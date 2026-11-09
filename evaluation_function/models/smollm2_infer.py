"""
SmolLM2-135M-Instruct inference using the pre-trained Hugging Face checkpoint.

Shown alongside ``gpt2_infer`` for comparison: a similar parameter count, but
newer training data, a newer (Llama-style) architecture and instruction tuning.
The student's response is wrapped in the model's chat template as a user
message, and only the model's reply is returned as feedback. Unlike GPT-2, the
prompt is not echoed: the student asked a question rather than starting a
sentence for the model to finish.

Weights come from ``SMOLLM2_MODEL_DIR`` when present, otherwise they are
downloaded on first use (see ``hf_lm``).

Params:
- ``word_count``: maximum number of new tokens to generate (default 64, max 256);
  the reply may end sooner when the model finishes its turn
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

MODEL = HFCausalLM(
    "SmolLM2",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    dir_env="SMOLLM2_MODEL_DIR",
    id_env="SMOLLM2_MODEL_ID",
)
DEFAULT_PROMPT = "What is a language model?"
DEFAULT_TOKENS = 64
MAX_TOKENS = 256
VOCAB_SIZE = 49152


def reply(prompt, max_new_tokens=DEFAULT_TOKENS, temperature=1.0, top_k=50, seed=None):
    model, tokenizer, device = MODEL.load()
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    output = sample(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, seed)
    return tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True).strip()


def run(response, answer, params: Params) -> Result:
    # Guard the whole load+infer path so a failed download or corrupt weights
    # degrade this one request rather than taking down the RPC worker.
    try:
        text = reply(
            prompt_text(response, DEFAULT_PROMPT),
            **sampling_params(params, DEFAULT_TOKENS, MAX_TOKENS, VOCAB_SIZE),
        )
    except Exception as e:
        return failure_result("smollm2_infer", "SmolLM2", e)

    result = Result(True)
    result.add_feedback("general", text)
    return result
