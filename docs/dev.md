# langModels
An evaluation function to aid the learning of language models.

**NOTE:** AWS deployments need at least 2048MB RAM, and storage to match the ~3.4GB image. GPT-2 (`gpt2_infer`) peaks near 1GB in the Python process alone, so 1024MB is no longer enough.

## Inputs
*Specific input parameters which can be supplied when the `eval` command is supplied to this function.*

The parameters for each model are documented for teachers in `user.md`.

## Models: build and runtime notes

### `gpt2_infer`
- `scripts/download_gpt2.py` downloads the weights into the image during `docker build`. They go to `GPT2_MODEL_DIR` (`/app/hf_models/gpt2`), so nothing is fetched at runtime.
- Locally, the weights download on first use into `HF_HOME` (default `/tmp/hf`).
- The test that does a real download and generation is skipped unless `RUN_GPT2_TESTS=1` is set, so CI doesn't download about 500MB.
- Memory: see the note above (2048MB minimum).

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