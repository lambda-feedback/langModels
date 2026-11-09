"""
GPT-2 (124M) inference using the pre-trained Hugging Face ``gpt2`` checkpoint.

The next rung on the ladder after the Bengio neural n-gram model: a full
transformer language model. The student's response is used as the prompt and
the model's sampled continuation (prompt included) is returned as feedback.

Weights come from ``GPT2_MODEL_DIR`` when present, otherwise they are downloaded
on first use (see ``hf_lm``).

Params:
- ``word_count``: number of new tokens to generate (default 20, max 100)
- ``temperature``: sampling temperature (default 1.0)
- ``top_k``: sample only from the k most likely tokens (default 50)
- ``seed``: optional integer for reproducible output
"""
from lf_toolkit.evaluation import Result, Params

from evaluation_function.models.hf_lm import (
    HFCausalLM,
    failure_result,
    prompt_text,
    sample,
    sampling_params,
)

MODEL = HFCausalLM("GPT-2", "gpt2", dir_env="GPT2_MODEL_DIR", id_env="GPT2_MODEL_ID")
DEFAULT_PROMPT = "the general"
DEFAULT_TOKENS = 20
MAX_TOKENS = 100
VOCAB_SIZE = 50257


def complete(prompt, max_new_tokens=DEFAULT_TOKENS, temperature=1.0, top_k=50, seed=None):
    model, tokenizer, device = MODEL.load()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = sample(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, seed)
    return tokenizer.decode(output, skip_special_tokens=True)


def run(response, answer, params: Params) -> Result:
    # Guard the whole load+infer path so a failed download or corrupt weights
    # degrade this one request rather than taking down the RPC worker.
    try:
        text = complete(
            prompt_text(response, DEFAULT_PROMPT),
            **sampling_params(params, DEFAULT_TOKENS, MAX_TOKENS, VOCAB_SIZE),
        )
    except Exception as e:
        return failure_result("gpt2_infer", "GPT-2", e)

    result = Result(True)
    result.add_feedback("general", text)
    return result
