"""Shared loading and sampling for the pre-trained Hugging Face language models.

Weights are baked into the Docker image at build time (see ``Dockerfile``) and
located via a per-model ``*_MODEL_DIR`` env var. When that directory is missing
(e.g. local development) the model is downloaded from Hugging Face on first use
into ``HF_HOME`` (default ``/tmp/hf``, the only writable path on Lambda).

Models are loaded in float32: Lambda is CPU-only, where half-precision is slow
and quantisation would distort the comparison between models.
"""
import os
import traceback
from pathlib import Path

from lf_toolkit.evaluation import Result

from evaluation_function.lazy_load import LazyModule

torch = LazyModule("torch")
transformers = LazyModule("transformers")


class HFCausalLM:
    """A pre-trained causal LM, loaded on first use and cached for the worker's lifetime."""

    def __init__(self, label, default_model_id, dir_env, id_env):
        self.label = label
        self.default_model_id = default_model_id
        self.dir_env = dir_env
        self.id_env = id_env
        self._loaded = None

    def load(self):
        """Return ``(model, tokenizer, device)``."""
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _load(self):
        model_dir = os.environ.get(self.dir_env)
        if model_dir and Path(model_dir).is_dir():
            print(f"### Loading {self.label} from {model_dir} ###")
            source, kwargs = model_dir, {"local_files_only": True}
        else:
            source = os.environ.get(self.id_env, self.default_model_id)
            cache_dir = os.environ.get("HF_HOME", "/tmp/hf")
            print(f"### Downloading {self.label} '{source}' into {cache_dir} ###")
            kwargs = {"cache_dir": cache_dir}

        tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
        model = transformers.AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, **kwargs)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device).eval()
        return model, tokenizer, device


def sample(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, seed=None):
    """Sample a continuation of ``input_ids``; returns the full token sequence (prompt included)."""
    if seed is not None:
        torch.manual_seed(int(seed))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
    return output[0]


def sampling_params(params, default_tokens, max_tokens, vocab_size):
    """Read the sampling params shared by all the Hugging Face models, clamped to valid ranges."""
    return {
        "max_new_tokens": _int_param(params, "word_count", default_tokens, 1, max_tokens),
        "temperature": _float_param(params, "temperature", 1.0, 0.01, 5.0),
        "top_k": _int_param(params, "top_k", 50, 1, vocab_size),
        "seed": params.get("seed"),
    }


def prompt_text(response, default):
    return response.strip() if isinstance(response, str) and response.strip() else default


def failure_result(name, label, exc) -> Result:
    """Turn a load or inference failure into a feedback result rather than crashing the worker."""
    print(f"### {name} failed ###")
    traceback.print_exc()
    return Result(
        is_correct=False,
        feedback_items=[
            ("general", f"Could not run the {label} language model."),
            ("error", str(exc)),
        ],
    )


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
