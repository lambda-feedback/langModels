"""Download the pre-trained GPT-2 weights and tokenizer into a local directory.

Used at Docker build time to bake the model into the image so that
``evaluation_function.models.gpt2_infer`` needs no network access at runtime.

Usage:
    python scripts/download_gpt2.py [TARGET_DIR] [--model-id gpt2]

TARGET_DIR defaults to $GPT2_MODEL_DIR.
"""
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=os.environ.get("GPT2_MODEL_DIR"),
        help="Directory to save the model into (default: $GPT2_MODEL_DIR)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GPT2_MODEL_ID", "gpt2"),
        help="Hugging Face model id (default: $GPT2_MODEL_ID or 'gpt2')",
    )
    args = parser.parse_args()

    if not args.target_dir:
        parser.error("no target directory given and GPT2_MODEL_DIR is not set")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading '{args.model_id}' into {args.target_dir}")
    AutoTokenizer.from_pretrained(args.model_id).save_pretrained(args.target_dir)
    AutoModelForCausalLM.from_pretrained(
        args.model_id, use_safetensors=True
    ).save_pretrained(args.target_dir)
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
