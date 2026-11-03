"""
GPT-2 (124M) inference using the pre-trained Hugging Face ``gpt2`` checkpoint.

The next rung on the ladder after the Bengio neural n-gram model: a full
transformer language model. The student's response is used as the prompt and
the model's sampled continuation is returned as feedback.

Weights are baked into the Docker image at build time (see ``Dockerfile``) and
located via the ``GPT2_MODEL_DIR`` env var. When that directory is missing
(e.g. local development) the model is downloaded from Hugging Face on first use
into ``HF_HOME`` (default ``/tmp/hf``, the only writable path on Lambda).

Params:
- ``word_count``: number of new tokens to generate (default 20, max 100)
- ``temperature``: sampling temperature (default 1.0)
- ``top_k``: sample only from the k most likely tokens (default 50)
- ``seed``: optional integer for reproducible output
"""
import os
import traceback
from pathlib import Path

from lf_toolkit.evaluation import Result, Params

from evaluation_function.lazy_load import LazyModule

torch = LazyModule("torch")
transformers = LazyModule("transformers")

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_PROMPT = "the general"
DEFAULT_TOKENS = 20
MAX_TOKENS = 100

# Cached across requests so a warm worker loads the weights only once.
_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _load():
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    model_dir = os.environ.get("GPT2_MODEL_DIR")
    if model_dir and Path(model_dir).is_dir():
        print(f"### Loading GPT-2 from {model_dir} ###")
        source, kwargs = model_dir, {"local_files_only": True}
    else:
        source = os.environ.get("GPT2_MODEL_ID", DEFAULT_MODEL_ID)
        cache_dir = os.environ.get("HF_HOME", "/tmp/hf")
        print(f"### Downloading GPT-2 '{source}' into {cache_dir} ###")
        kwargs = {"cache_dir": cache_dir}

    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
    model = transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    _MODEL, _TOKENIZER, _DEVICE = model, tokenizer, device
    return _MODEL, _TOKENIZER, _DEVICE


def _int_param(params, name, default, lo, hi):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _float_param(params, name, default, lo, hi):
    try:
        value = float(params.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def complete(prompt, max_new_tokens=DEFAULT_TOKENS, temperature=1.0, top_k=50, seed=None):
    model, tokenizer, device = _load()
    if seed is not None:
        torch.manual_seed(int(seed))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run(response, answer, params: Params) -> Result:
    # Guard the whole load+infer path so a failed download or corrupt weights
    # degrade this one request rather than taking down the RPC worker.
    try:
        return _run_inference(response, params)
    except Exception as e:
        print("### gpt2_infer failed ###")
        traceback.print_exc()
        return Result(
            is_correct=False,
            feedback_items=[
                ("general", "Could not run the GPT-2 language model."),
                ("error", str(e)),
            ],
        )


def _run_inference(response, params: Params) -> Result:
    prompt = response.strip() if isinstance(response, str) and response.strip() else DEFAULT_PROMPT

    text = complete(
        prompt,
        max_new_tokens=_int_param(params, "word_count", DEFAULT_TOKENS, 1, MAX_TOKENS),
        temperature=_float_param(params, "temperature", 1.0, 0.01, 5.0),
        top_k=_int_param(params, "top_k", 50, 1, 50257),
        seed=params.get("seed"),
    )

    result = Result(True)
    result.add_feedback("general", text)
    return result
