import sys, argparse, json, os
from pathlib import Path

from lf_toolkit.shared.params import Params

from .evaluation import evaluation_function

def dev():
    """Run the evaluation function from the command line for development purposes.

    Usage: 
    poetry run python -m evaluation_function.dev --config configs/dev.json --case basic_nn

    (Change the case as desired, and ensure the dev.json is up to date with your needs)

    """

    BASE_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--case", help="Case inside the config file")
    args = parser.parse_args()

    config_path = (BASE_DIR / args.config).resolve()
    with open(config_path) as f:
        all_config = json.load(f)

    if args.case not in all_config: # extract config for the relevant case
        raise ValueError(f"Case '{args.case}' not found in {args.config}")

    config = all_config[args.case]

    # Separate out required fields
    answer = config.pop("answer")
    response = config.pop("response")
    params = Params(**config)

    result = evaluation_function(answer, response, params)
    print(result.to_dict())

if __name__ == "__main__":
    dev()