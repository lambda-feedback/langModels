"""Download a pre-trained Hugging Face causal LM and its tokenizer into a local directory.

Used at Docker build time to bake the models into the image so that the
Hugging Face models (see ``evaluation_function/models/hf_lm.py``) need no
network access at runtime. Weights are saved in float32, the precision they are served in.

Usage:
    python scripts/download_hf_model.py TARGET_DIR --model-id MODEL_ID

e.g.
    python scripts/download_hf_model.py /app/hf_models/gpt2 --model-id gpt2
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target_dir", help="Directory to save the model into")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id, e.g. 'gpt2'")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading '{args.model_id}' into {args.target_dir}")
    AutoTokenizer.from_pretrained(args.model_id).save_pretrained(args.target_dir)
    AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32, use_safetensors=True
    ).save_pretrained(args.target_dir)
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
