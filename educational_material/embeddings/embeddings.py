"""Do directions in embedding space carry meaning?

Experiments on the token embeddings of the course's three neural models:
Bengio (Q5), GPT-2 (Q6) and SmolLM2 (Q7). Writes every table to results.md and
the figures to plots/, next to this file.

    poetry run python -m educational_material.embeddings.embeddings

GPT-2 and SmolLM2 download from Hugging Face on first run (into $HF_HOME, default /tmp/hf).
"""
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent.parent / "evaluation_function" / "models" / "storage"
PLOTS = HERE / "plots"

FROM_COLOUR, TO_COLOUR, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"

NEIGHBOUR_WORDS = ["king", "woman", "dog", "house", "red", "happy", "walked", "three", "France", "Monday", "coffee"]

# (x, y) pairs where y = x + relation. Analogies take every ordered pair of pairs:
# y2 ≈ x2 - x1 + y1, e.g. queen ≈ king - man + woman.
RELATIONS = {
    "plural": [("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("house", "houses"), ("book", "books"),
               ("tree", "trees"), ("bird", "birds"), ("hand", "hands"), ("girl", "girls"), ("door", "doors")],
    "past tense": [("walk", "walked"), ("play", "played"), ("look", "looked"), ("open", "opened"), ("go", "went"),
                   ("see", "saw"), ("run", "ran"), ("eat", "ate"), ("take", "took"), ("come", "came")],
    "comparative": [("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("old", "older"),
                    ("young", "younger"), ("tall", "taller"), ("good", "better"), ("bad", "worse")],
    "gender": [("man", "woman"), ("king", "queen"), ("uncle", "aunt"), ("actor", "actress"), ("boy", "girl"),
               ("brother", "sister"), ("father", "mother"), ("son", "daughter"), ("husband", "wife"),
               ("he", "she"), ("prince", "princess"), ("Mr", "Mrs")],
    "number → ordinal": [("one", "first"), ("two", "second"), ("three", "third"), ("four", "fourth"),
                         ("five", "fifth"), ("six", "sixth"), ("seven", "seventh"), ("ten", "tenth")],
    "country → capital": [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Japan", "Tokyo"),
                          ("Spain", "Madrid"), ("England", "London"), ("Russia", "Moscow"), ("China", "Beijing"),
                          ("Egypt", "Cairo"), ("Greece", "Athens")],
    "country → nationality": [("France", "French"), ("Germany", "German"), ("Italy", "Italian"),
                              ("Japan", "Japanese"), ("Spain", "Spanish"), ("China", "Chinese"),
                              ("Russia", "Russian"), ("Greece", "Greek"), ("England", "English")],
    "country → food/drink": [("Italy", "pizza"), ("Japan", "sushi"), ("Germany", "beer"), ("France", "wine"),
                             ("Mexico", "tacos"), ("India", "curry"), ("Russia", "vodka")],
}

SHOWCASE = [("king", "man", "woman", "queen"), ("uncle", "man", "woman", "aunt"),
            ("actor", "man", "woman", "actress"), ("cats", "cat", "dog", "dogs"),
            ("walked", "walk", "play", "played"), ("bigger", "big", "small", "smaller"),
            ("Paris", "France", "Germany", "Berlin"), ("French", "France", "Italy", "Italian"),
            ("pizza", "Italy", "Japan", "sushi"), ("first", "one", "two", "second")]

# Directions are averaged over the first pairs and tested on held-out ones.
DIRECTIONS = {
    "plural": dict(
        train=RELATIONS["plural"][:5],
        test=RELATIONS["plural"][5:],
        probes=["one", "single", "two", "three", "several", "many", "hundred"],
    ),
    "gender": dict(
        train=[("man", "woman"), ("he", "she"), ("boy", "girl"), ("father", "mother"), ("son", "daughter")],
        test=[("king", "queen"), ("uncle", "aunt"), ("actor", "actress"), ("brother", "sister"),
              ("husband", "wife")],
        probes=["nurse", "doctor", "teacher", "engineer", "secretary", "soldier", "dancer", "programmer"],
    ),
    "country → capital": dict(
        train=RELATIONS["country → capital"][:5],
        test=RELATIONS["country → capital"][5:],
        probes=[],
    ),
}

PCA_SETS = {
    "gender": [("man", "woman"), ("king", "queen"), ("uncle", "aunt"), ("boy", "girl"), ("brother", "sister")],
    "plural": [("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("house", "houses"), ("book", "books")],
    "country → capital": [("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"), ("Spain", "Madrid"),
                          ("Russia", "Moscow")],
}


def normalise(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(1e-8)


@dataclass
class Space:
    """A model's token embedding matrix, restricted to whole-word tokens for lookups."""

    name: str
    vectors: np.ndarray
    pieces: list
    prefix: str  # marks a word-initial token: "▁" (SentencePiece) or "Ġ" (byte-level BPE)
    lowercase: bool  # Bengio's corpus was lowercased
    unit: dict = field(init=False)

    def __post_init__(self):
        self.ids = {p: i for i, p in enumerate(self.pieces)}
        n = len(self.prefix)
        self.word_ids = np.array([i for i, p in enumerate(self.pieces)
                                  if p.startswith(self.prefix) and p[n:].isascii() and p[n:].isalpha()])
        self.unit = {False: normalise(self.vectors), True: normalise(self.vectors - self.vectors.mean(0))}
        self.word_unit = {c: m[self.word_ids] for c, m in self.unit.items()}

    def id(self, word):
        return self.ids.get(self.prefix + (word.lower() if self.lowercase else word))

    def has(self, *words):
        return all(self.id(w) is not None for w in words)

    def word(self, i):
        return self.pieces[i][len(self.prefix):]

    def vec(self, word, centred):
        return self.unit[centred][self.id(word)]

    def nearest(self, target, exclude, k, centred):
        sims = self.word_unit[centred] @ normalise(target)
        skip = {w.lower() for w in exclude}
        top = np.argpartition(-sims, k + 30)[:k + 30]
        out = []
        for j in top[np.argsort(-sims[top])]:
            w = self.word(self.word_ids[j])
            if w.lower() not in skip:
                out.append(w)
                skip.add(w.lower())  # drop case variants: " King" after " king"
                if len(out) == k:
                    break
        return out

    def neighbours(self, word, k=8, centred=True):
        return self.nearest(self.vec(word, centred), [word], k, centred)

    def analogy(self, a, b, c, k=5, centred=True):
        """a - b + c, e.g. king - man + woman."""
        target = self.vec(a, centred) - self.vec(b, centred) + self.vec(c, centred)
        return self.nearest(target, [a, b, c], k, centred)

    def direction(self, pairs, centred=True):
        pairs = [p for p in pairs if self.has(*p)]
        return normalise(np.mean([self.vec(y, centred) - self.vec(x, centred) for x, y in pairs], axis=0))


def load_bengio():
    import sentencepiece as spm
    import torch

    weights = torch.load(STORAGE / "bengio_model.pt", map_location="cpu")["emb.weight"].numpy()
    sp = spm.SentencePieceProcessor(model_file=str(STORAGE / "bpe.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    return Space("Bengio", weights, pieces, prefix="▁", lowercase=True)


def load_hf(name, model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = os.environ.get("HF_HOME", "/tmp/hf")
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
    weights = model.get_input_embeddings().weight.detach().float().numpy()
    pieces = tokenizer.convert_ids_to_tokens(list(range(weights.shape[0])))
    return Space(name, weights, pieces, prefix="Ġ", lowercase=False)


# ---------------------------------------------------------------- experiments

def vocab_table(spaces):
    words = sorted({w for pairs in RELATIONS.values() for p in pairs for w in p} | set(NEIGHBOUR_WORDS),
                   key=str.lower)
    lines = ["| model | vocab | whole-word tokens | test words found | missing |", "|---|---|---|---|---|"]
    for s in spaces:
        missing = [w for w in words if not s.has(w)]
        lines.append(f"| {s.name} | {len(s.pieces)} × {s.vectors.shape[1]} | {len(s.word_ids)} | "
                     f"{len(words) - len(missing)}/{len(words)} | {', '.join(missing) or '-'} |")
    return lines


def neighbour_tables(spaces):
    lines = []
    for centred in (True, False):
        lines += [f"\n### Nearest neighbours ({'centred' if centred else 'not centred'})\n",
                  "| word | " + " | ".join(s.name for s in spaces) + " |", "|---" * (len(spaces) + 1) + "|"]
        for w in NEIGHBOUR_WORDS:
            cells = [", ".join(s.neighbours(w, centred=centred)) if s.has(w) else "-" for s in spaces]
            lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return lines


def analogy_scores(space, pairs, centred):
    """Top-1 and top-5 hits over all ordered pairs of pairs, plus a baseline that ignores the offset."""
    pairs = [p for p in pairs if space.has(*p)]
    top1 = top5 = base5 = n = 0
    for (x1, y1), (x2, y2) in itertools.permutations(pairs, 2):
        if len({x1, y1, x2, y2}) < 4:
            continue
        guesses = space.analogy(x2, x1, y1, centred=centred)
        top1 += guesses[0] == y2
        top5 += y2 in guesses
        base5 += y2 in space.neighbours(x2, k=5, centred=centred)
        n += 1
    return top1, top5, base5, n


def analogy_tables(spaces):
    lines = []
    for centred in (True, False):
        lines += [f"\n### Analogy accuracy ({'centred' if centred else 'not centred'})\n",
                  "Each cell: top-1 % / top-5 % (baseline top-5 %: nearest neighbours of x2 alone, no offset)"
                  " [number of analogies].\n",
                  "| relation | " + " | ".join(s.name for s in spaces) + " |", "|---" * (len(spaces) + 1) + "|"]
        for rel, pairs in RELATIONS.items():
            cells = []
            for s in spaces:
                t1, t5, b5, n = analogy_scores(s, pairs, centred)
                cells.append(f"{100 * t1 / n:.0f} / {100 * t5 / n:.0f} ({100 * b5 / n:.0f}) [{n}]" if n else "-")
            lines.append(f"| {rel} | " + " | ".join(cells) + " |")
    lines += ["\n### Showcase analogies (centred, top 5; ✓ = expected answer ranked first)\n",
              "| analogy | expected | " + " | ".join(s.name for s in spaces) + " |",
              "|---" * (len(spaces) + 2) + "|"]
    for a, b, c, want in SHOWCASE:
        cells = []
        for s in spaces:
            if not s.has(a, b, c, want):
                cells.append("-")
                continue
            got = s.analogy(a, b, c)
            cells.append(("✓ " if got[0].lower() == want.lower() else "") + ", ".join(got))
        lines.append(f"| {a} - {b} + {c} | {want} | " + " | ".join(cells) + " |")
    return lines


def direction_tables(spaces):
    lines = []
    for name, d in DIRECTIONS.items():
        lines += [f"\n### Direction: {name}\n",
                  f"Built from {', '.join(f'{y} - {x}' for x, y in d['train'])}. "
                  "Held-out: share of pairs where the second word projects further along the direction.\n",
                  "| model | centring | held-out pairs ordered | probe projections (high → low) |", "|---|---|---|---|"]
        for s in spaces:
            for centred in (True, False):
                u = s.direction(d["train"], centred)
                test = [p for p in d["test"] if s.has(*p)]
                ok = sum(s.vec(y, centred) @ u > s.vec(x, centred) @ u for x, y in test)
                probes = sorted(((s.vec(w, centred) @ u, w) for w in d["probes"] if s.has(w)), reverse=True)
                lines.append(f"| {s.name} | {'centred' if centred else 'raw'} | {ok}/{len(test)} | "
                             + (", ".join(f"{w} {v:+.2f}" for v, w in probes) or "-") + " |")
    return lines


# ---------------------------------------------------------------------- plots

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def draw_pairs(ax, space, pairs, P):
    for i, (x0, y0) in enumerate(P[0::2]):
        x1, y1 = P[2 * i + 1]
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    colours = [FROM_COLOUR, TO_COLOUR] * len(pairs)
    for (px, py), word, colour, dy in zip(P, [w for p in pairs for w in p], colours, label_offsets(P)):
        ax.plot(px, py, "o", ms=8, color=colour, mec=SURFACE, mew=2)
        ax.annotate(space.word(space.id(word)), (px, py), (0, dy), textcoords="offset points", ha="center",
                    va="center", fontsize=9, color=INK)


def label_offsets(P):
    """Put a label above its point, or below if that would collide with a label already placed."""
    span = np.ptp(P, axis=0).clip(1e-9)
    placed, offsets = [], []
    for point in P / span:
        for dy in (10, -11, 21, -22):
            spot = point + [0, dy / 300]
            if all(abs(spot[0] - q[0]) > 0.12 or abs(spot[1] - q[1]) > 0.05 for q in placed):
                break
        placed.append(spot)
        offsets.append(dy)
    return offsets


def pair_panels(spaces, title, filename, panel):
    """One panel per model, side by side; `panel` returns the 2D points for a model's pairs, or None."""
    fig, axes = plt.subplots(1, len(spaces), figsize=(4.2 * len(spaces), 4), facecolor=SURFACE)
    for ax, s in zip(axes, spaces):
        style(ax)
        ax.set_title(s.name, color=INK, fontsize=11)
        ax.set_yticks([])
        drawn = panel(ax, s)
        if drawn is None:
            ax.set_xticks([])
            ax.text(0.5, 0.5, "too few of these words\nin the vocabulary", transform=ax.transAxes, ha="center",
                    va="center", color=MUTED, fontsize=9)
        else:
            draw_pairs(ax, s, *drawn)
            ax.margins(0.18)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pca(spaces, name, pairs):
    """Plain 2D PCA of the pairs' embeddings, oriented so the mean arrow points right."""
    def panel(ax, s):
        found = [p for p in pairs if s.has(*p)]
        if len(found) < 3:
            return None
        X = np.array([s.vec(w, True) for p in found for w in p])
        X -= X.mean(0)
        P = X @ np.linalg.svd(X, full_matrices=False)[2][:2].T
        if (P[1::2] - P[0::2]).mean(0)[0] < 0:
            P[:, 0] *= -1
        ax.set_xticks([])
        return found, P

    pair_panels(spaces, f"{name}: 2D PCA of the word embeddings", f"pca_{slug(name)}.png", panel)


def plot_direction_view(spaces, name, d):
    """Held-out pairs placed along a direction learned from other pairs (x) against the main remaining
    variation (y, PCA). Arrows pointing right mean the direction generalises to new words."""
    def panel(ax, s):
        test = [p for p in d["test"] if s.has(*p)]
        if len(test) < 3 or not any(s.has(*p) for p in d["train"]):
            return None
        u = s.direction(d["train"])
        X = np.array([s.vec(w, True) for p in test for w in p])
        x = X @ u
        rest = X - np.outer(x, u)
        rest -= rest.mean(0)
        y = rest @ np.linalg.svd(rest, full_matrices=False)[2][0]
        ax.axvline(0, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel(f"along the {name} direction", color=MUTED, fontsize=9)
        return test, np.column_stack([x, y])

    pair_panels(spaces, f"{name}: held-out pairs along the direction learned from other pairs",
                f"direction_view_{slug(name)}.png", panel)


def plot_direction(spaces, name, d, labels):
    """Each word's position along a direction: held-out pairs on two rows, probe words on a third."""
    fig, axes = plt.subplots(len(spaces), 1, figsize=(9, 2.6 * len(spaces)), facecolor=SURFACE)
    rows = [labels[0], labels[1], "probe words"]
    for ax, s in zip(axes, spaces):
        u = s.direction(d["train"])
        test = [p for p in d["test"] if s.has(*p)]
        groups = [[x for x, _ in test], [y for _, y in test], [w for w in d["probes"] if s.has(w)]]
        style(ax)
        for row, (words, colour) in enumerate(zip(groups, (FROM_COLOUR, TO_COLOUR, MUTED))):
            proj = [s.vec(w, True) @ u for w in words]
            ax.plot(proj, [row] * len(proj), "o", ms=8, color=colour, mec=SURFACE, mew=2)
            for i, (p, w) in enumerate(sorted(zip(proj, words))):
                ax.annotate(s.word(s.id(w)), (p, row), (0, (8, -14, 18)[i % 3]), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
        ax.set_yticks(range(3), rows, fontsize=9, color=INK)
        ax.set_ylim(-0.9, 3.0)
        ax.set_title(s.name, loc="left", color=INK, fontsize=10)
    axes[-1].set_xlabel(f"projection onto the {name} direction (cosine)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / f"direction_{slug(name)}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_").replace("___", "_")


def main():
    spaces = [load_bengio(), load_hf("GPT-2", "gpt2"), load_hf("SmolLM2", "HuggingFaceTB/SmolLM2-135M-Instruct")]

    lines = ["# Embedding experiments: raw results", "",
             "Generated by `embeddings.py`; see `report.md` for the interpretation. Vectors are unit-normalised; "
             "'centred' means the mean embedding was subtracted first. Neighbour searches only consider whole-word "
             "tokens and exclude the input words.", "", "### Vocabulary coverage", ""]
    lines += vocab_table(spaces) + neighbour_tables(spaces) + analogy_tables(spaces) + direction_tables(spaces)
    (HERE / "results.md").write_text("\n".join(lines) + "\n")

    PLOTS.mkdir(exist_ok=True)
    for name, pairs in PCA_SETS.items():
        plot_pca(spaces, name, pairs)
        plot_direction_view(spaces, name, DIRECTIONS[name])
    plot_direction(spaces, "plural", DIRECTIONS["plural"], ("singular", "plural"))
    plot_direction(spaces, "gender", DIRECTIONS["gender"], ("male", "female"))
    print(f"Wrote {HERE / 'results.md'} and {PLOTS}/")


if __name__ == "__main__":
    main()
