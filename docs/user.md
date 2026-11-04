# langModels

A series of language models that students can run for inference to explore their performance.

Created from scratch with just a neural network for modeling data from a sine wave. This is to ensure the software all works in integration tests. Following that, more models will be added (Shannon ngrams (letters); Shannon ngrams (words); Bengio neural network model of language with a small context window; basic transformer models; larger transformer models; external LLMs).

## Models

### `gpt2_infer`
Pre-trained GPT-2 (124M) from Hugging Face. The student's response is used as the prompt, and GPT-2's sampled continuation is returned as feedback.

| Param | Default | Notes |
|---|---|---|
| `word_count` | 20 | New tokens to generate (1–100) |
| `temperature` | 1.0 | Sampling temperature (0.01–5.0): higher values give more varied text |
| `top_k` | 50 | Sample only from the k most likely next tokens (1–50257) |
| `seed` | none | Integer for reproducible output |

### `smollm2_infer`
Pre-trained SmolLM2-135M-Instruct from Hugging Face, for comparison with GPT-2: a similar size (135M parameters), but trained on newer data, with a newer architecture, and instruction-tuned to act as a chat assistant. The student's response is sent as a chat message, and only the model's reply is returned. Unlike GPT-2, the prompt is not repeated at the start of the feedback. The model's chat template adds a default system prompt ("You are a helpful AI assistant named SmolLM, trained by Hugging Face"), so replies may refer to the model as SmolLM.

| Param | Default | Notes |
|---|---|---|
| `word_count` | 64 | Maximum new tokens to generate (1–256). The reply can be shorter if the model finishes its answer |
| `temperature` | 1.0 | Sampling temperature (0.01–5.0): higher values give more varied text |
| `top_k` | 50 | Sample only from the k most likely next tokens (1–49152) |
| `seed` | none | Integer for reproducible output |
