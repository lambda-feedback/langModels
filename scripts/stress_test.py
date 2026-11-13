#!/usr/bin/env python3
"""Stress test the langModels /evaluate HTTP endpoint.

Requests are built from the response areas of a Lambda Feedback set export
(default: set_Context_engineering/), so the load mirrors what students send.

Stdlib only, so it runs with any Python 3.11+ without poetry:

    python3 scripts/stress_test.py --dry-run
    python3 scripts/stress_test.py --requests 200 --concurrency 10
    python3 scripts/stress_test.py --ramp 1,5,10,25,50 --duration 30 --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

DEFAULT_URL = (
    "https://y8ca0qezcc.execute-api.eu-west-2.amazonaws.com"
    "/default/langModels-staging/evaluate"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SET_DIR = os.path.join(REPO_ROOT, "set_Context_engineering")


@dataclass
class Case:
    label: str
    model: str
    body: bytes


@dataclass
class Sample:
    stage: str
    case: str
    model: str
    status: int  # 0 when no HTTP response was received
    outcome: str  # ok | http_<code> | error_in_body | bad_json | timeout | conn_error
    latency_s: float
    size: int
    detail: str = ""


def load_cases(set_dir: str, models: list[str] | None) -> list[Case]:
    cases = []
    for path in sorted(glob.glob(os.path.join(set_dir, "question_*.json"))):
        with open(path) as f:
            question = json.load(f)
        qnum = os.path.basename(path).split("_")[1]
        title = question.get("title", "")
        for part in question.get("parts", []):
            for area in part.get("responseAreas", []):
                if area.get("evaluationFunctionName") != "langModels":
                    continue
                params = area.get("gradeParams") or {}
                model = params.get("model", "basic_nn")
                if models and model not in models:
                    continue
                answer = area.get("response", {}).get("responseInput", {}).get("answer")
                order = area.get("orderNumber", 0)
                body = {
                    "submission": {"type": "OTHER", "format": "", "content": {"value": answer}},
                    "task": {
                        "title": f"{title} / area {order}",
                        "referenceSolution": {"value": answer},
                    },
                    "configuration": {"params": params},
                }
                cases.append(Case(
                    label=f"q{qnum}.a{order}",
                    model=model,
                    body=json.dumps(body).encode(),
                ))
    return cases


SYNTHETIC_PROMPTS = {
    "smollm2_infer": [
        "What is a language model?",
        "Explain attention in transformers in one sentence.",
        "Write a haiku about tokens.",
        "Why do large language models sometimes make things up?",
        "What is the difference between a bigram and a trigram?",
    ],
}


def synthetic_cases(models: list[str], word_counts: list[int]) -> list[Case]:
    """Cases for models with no response area in the set export, shaped like
    {"response": prompt, "params": {"model": ..., "word_count": N}}. Case.model
    carries the length so report() breaks latency down per word_count.
    Cases interleave lengths so every stage sees a mix."""
    cases = []
    for model in models:
        prompts = SYNTHETIC_PROMPTS.get(model)
        if not prompts:
            raise SystemExit(f"no synthetic prompts for {model!r}; known: {', '.join(SYNTHETIC_PROMPTS)}")
        short = model.removesuffix("_infer")
        for i, prompt in enumerate(prompts):
            for wc in word_counts:
                body = {
                    "submission": {"type": "OTHER", "format": "", "content": {"value": prompt}},
                    "task": {"title": f"synthetic {model}", "referenceSolution": {"value": ""}},
                    "configuration": {"params": {"model": model, "word_count": wc}},
                }
                cases.append(Case(
                    label=f"{short}.w{wc}.p{i}",
                    model=f"{model}/w{wc}",
                    body=json.dumps(body).encode(),
                ))
    return cases


def is_success_body(data) -> bool:
    """The endpoint returns a JSON list of feedback items on success, e.g.
    [{"awardedPoints": 1, "message": "...", ...}]. Also accept the toolkit's
    {"result": ...} envelope; anything with an "error" key is a failure."""
    if isinstance(data, list):
        return bool(data) and all(isinstance(i, dict) and "error" not in i for i in data)
    if isinstance(data, dict):
        return "error" not in data and ("result" in data or "awardedPoints" in data)
    return False


def send(url: str, case: Case, timeout: float, stage: str) -> Sample:
    req = urllib.request.Request(
        url, data=case.body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    status, outcome, size, detail = 0, "ok", 0, ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read() or b""
        outcome = f"http_{e.code}"
        detail = raw[:200].decode(errors="replace")
    except (TimeoutError, socket.timeout):
        raw, outcome, detail = b"", "timeout", f">{timeout}s"
    except urllib.error.URLError as e:
        raw = b""
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            outcome, detail = "timeout", f">{timeout}s"
        else:
            outcome, detail = "conn_error", str(e.reason)[:200]
    except (ConnectionError, OSError) as e:
        raw, outcome, detail = b"", "conn_error", str(e)[:200]
    latency = time.perf_counter() - start
    size = len(raw)

    if outcome == "ok":
        try:
            data = json.loads(raw)
            if not is_success_body(data):
                outcome = "error_in_body"
                detail = json.dumps(data)[:200]
        except ValueError:
            outcome, detail = "bad_json", raw[:200].decode(errors="replace")

    return Sample(stage, case.label, case.model, status, outcome, latency, size, detail)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def run_stage(url, cases, concurrency, n_requests, duration, timeout, stage) -> tuple[list[Sample], float]:
    samples: list[Sample] = []
    lock = threading.Lock()
    cycle = itertools.cycle(cases)
    counter = itertools.count()
    deadline = time.monotonic() + duration if duration else None

    def next_case():
        with lock:
            i = next(counter)
            if deadline is None and i >= n_requests:
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            return next(cycle)

    def worker():
        while (case := next_case()) is not None:
            s = send(url, case, timeout, stage)
            with lock:
                samples.append(s)
                done = len(samples)
            if done % 25 == 0:
                print(f"  [{stage}] {done} requests done", file=sys.stderr)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    return samples, time.perf_counter() - start


def report(title: str, samples: list[Sample], elapsed: float | None) -> None:
    print(f"\n=== {title} ===")
    if not samples:
        print("  (no samples)")
        return
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.model].append(s)
    groups["ALL"] = samples

    header = f"{'model':<24}{'n':>6}{'ok%':>7}{'min':>8}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    for model, group in groups.items():
        lat = sorted(s.latency_s for s in group)
        ok = sum(s.outcome == "ok" for s in group)
        print(
            f"{model:<24}{len(group):>6}{100 * ok / len(group):>6.1f}%"
            + "".join(f"{v:>8.2f}" for v in (
                lat[0], percentile(lat, 50), percentile(lat, 90),
                percentile(lat, 95), percentile(lat, 99), lat[-1],
            ))
        )
    print("(latencies in seconds)")
    if elapsed:
        print(f"throughput: {len(samples) / elapsed:.2f} req/s over {elapsed:.1f}s")

    failures = [s for s in samples if s.outcome != "ok"]
    if failures:
        print("failures:")
        first: dict[tuple[str, str], str] = {}
        counts = Counter((s.model, s.outcome) for s in failures)
        for s in failures:
            first.setdefault((s.model, s.outcome), s.detail)
        for (model, outcome), n in counts.most_common():
            print(f"  {model:<24} {outcome:<14} x{n:<5} e.g. {first[(model, outcome)]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="evaluate endpoint (default: staging)")
    ap.add_argument("--set-dir", default=DEFAULT_SET_DIR, help="set export directory with question_*.json")
    ap.add_argument("--concurrency", type=int, default=10, help="parallel workers (ignored with --ramp)")
    ap.add_argument("--requests", type=int, default=200, help="total requests per stage")
    ap.add_argument("--duration", type=float, help="run each stage for N seconds instead of --requests")
    ap.add_argument("--ramp", help="comma-separated concurrency stages, e.g. 1,5,10,25,50")
    ap.add_argument("--model", action="append", help="only use cases for this model (repeatable)")
    ap.add_argument("--synthetic", action="append",
                    help="add built-in prompt cases for a model not in the set export, e.g. smollm2_infer "
                         "(repeatable; set-export cases are then only included if --model is given)")
    ap.add_argument("--word-counts", default="16,64,256",
                    help="word_count values for --synthetic cases (default: 16,64,256)")
    ap.add_argument("--timeout", type=float, default=60, help="per-request timeout in seconds")
    ap.add_argument("--no-warmup", action="store_true", help="skip the sequential warm-up pass")
    ap.add_argument("--dry-run", action="store_true", help="print request bodies and exit")
    ap.add_argument("--out", help="write per-request results to this CSV")
    args = ap.parse_args()

    cases = []
    if args.model or not args.synthetic:
        cases += load_cases(args.set_dir, args.model)
    if args.synthetic:
        word_counts = [int(x) for x in args.word_counts.split(",")]
        cases += synthetic_cases(args.synthetic, word_counts)
    if not cases:
        print(f"No langModels cases found in {args.set_dir}", file=sys.stderr)
        return 1

    print(f"{len(cases)} cases from {args.set_dir}: "
          + ", ".join(f"{m}×{n}" for m, n in Counter(c.model for c in cases).items()))
    if args.dry_run:
        for c in cases:
            print(f"\n# {c.label} ({c.model})\n{json.dumps(json.loads(c.body), indent=2)}")
        return 0

    print(f"target: {args.url}")
    all_samples: list[Sample] = []

    if not args.no_warmup:
        warm = [send(args.url, c, args.timeout, "warmup") for c in cases]
        for s in warm:
            print(f"  warmup {s.case:<8} {s.model:<24} {s.outcome:<14} {s.latency_s:6.2f}s")
        report("warm-up (sequential, includes cold starts)", warm, None)
        all_samples += warm

    stages = [int(x) for x in args.ramp.split(",")] if args.ramp else [args.concurrency]
    for conc in stages:
        stage = f"c{conc}"
        what = f"{args.duration:g}s" if args.duration else f"{args.requests} requests"
        print(f"\n--> stage {stage}: concurrency {conc}, {what}", file=sys.stderr)
        samples, elapsed = run_stage(
            args.url, cases, conc, args.requests, args.duration, args.timeout, stage,
        )
        report(f"stage {stage} (concurrency {conc})", samples, elapsed)
        all_samples += samples

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_samples[0]).keys()))
            w.writeheader()
            w.writerows(asdict(s) for s in all_samples)
        print(f"\nwrote {len(all_samples)} rows to {args.out}")

    return 0 if all(s.outcome == "ok" for s in all_samples) else 2


if __name__ == "__main__":
    sys.exit(main())
