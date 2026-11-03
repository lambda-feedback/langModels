# langModels
An evaluation function to aid the learning of language models.

**NOTE: **  All AWS deployments should allocate 1024MB RAM and Storage due to the image size.

## Inputs
*Specific input parameters which can be supplied when the `eval` command is supplied to this function.*

### `gpt2_infer`
Pre-trained GPT-2 (124M) from Hugging Face. The student's `response` is the prompt; the sampled continuation is returned as `general` feedback.

| Param | Default | Notes |
|---|---|---|
| `word_count` | 20 | New tokens to generate (1–100) |
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Sample from the k most likely tokens |
| `seed` | none | Integer for reproducible output |

Weights are baked into the image at `GPT2_MODEL_DIR` during `docker build`. Locally, they are downloaded on first use into `HF_HOME` (default `/tmp/hf`). GPT-2 plus torch needs roughly 1–1.5 GB of RAM, so give the Lambda at least 2048MB if this model is enabled.

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