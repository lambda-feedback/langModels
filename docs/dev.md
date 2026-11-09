# langModels
An evaluation function to aid the learning of language models.

**NOTE:** AWS deployments need at least 2048MB RAM, and storage to match the ~4GB image. GPT-2 (`gpt2_infer`) peaks near 1GB in the Python process alone, so 1024MB is no longer enough. A warm worker that has served both GPT-2 and SmolLM2 (`smollm2_infer`) keeps both loaded: peak memory measured in the container was about 1.5GB, which leaves around 500MB of headroom at 2048MB.

## Inputs
*Specific input parameters which can be supplied when the `eval` command is supplied to this function.*

The parameters for each model are documented for teachers in `user.md`.

## Models: build and runtime notes

### Hugging Face models: `gpt2_infer` and `smollm2_infer`
Both share their loading and sampling code (`evaluation_function/models/hf_lm.py`), and are loaded in float32 with no quantisation, since Lambda runs on CPU.
- `scripts/download_hf_model.py` downloads the weights into the image during `docker build`, one layer per model. They go to `GPT2_MODEL_DIR` (`/app/hf_models/gpt2`) and `SMOLLM2_MODEL_DIR` (`/app/hf_models/smollm2-135m-instruct`), so nothing is fetched at runtime.
- Locally, the weights download on first use into `HF_HOME` (default `/tmp/hf`). Set `GPT2_MODEL_DIR` or `SMOLLM2_MODEL_DIR` to use a directory made by the download script instead.
- The tests that do a real download and generation are skipped unless `RUN_HF_MODEL_TESTS=1` is set, so CI doesn't download about 1GB.
- Memory: see the note above.

## Outputs
*Output schema/values for this function*

## Examples
*List of example inputs and outputs for this function, each under a different sub-heading*

### Simple Evaluation

```python
{
  "example": {
    "Something": "something"
  }
}
```

```python
{
  "example": {
    "Something": "something"
  }
}
```
## Stress testing

`scripts/stress_test.py` load-tests a deployed `/evaluate` endpoint. It uses the
response areas in a set export (default `set_Context_engineering/`), so the requests
use the same models and `gradeParams` that students trigger. It only needs the
standard library.

```bash
# Print the request bodies built from the set, without sending anything
python3 scripts/stress_test.py --dry-run

# 200 requests at concurrency 10 against staging (the default URL)
python3 scripts/stress_test.py --requests 200 --concurrency 10

# Step up the concurrency, 30s per stage, and save per-request results
python3 scripts/stress_test.py --ramp 1,5,10,25,50 --duration 30 --out results.csv

# Only one model, against another endpoint
python3 scripts/stress_test.py --model bengio_infer --url https://.../evaluate
```

Each run starts with a sequential warm-up pass, one request per case, so cold starts
are reported apart from the load stages. Each stage reports success rate and
min/p50/p90/p95/p99/max latency per model, plus throughput, and groups failures by
type (`http_<code>`, `timeout`, `conn_error`, `error_in_body`, `bad_json`). The exit
code is non-zero if any request failed.
