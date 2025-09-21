import sys

from lf_toolkit.shared.params import Params

from .evaluation import evaluation_function

def dev():
    """Run the evaluation function from the command line for development purposes.

    Usage: python -m evaluation_function.dev <answer> <response>
    """
    if len(sys.argv) < 3:
        print("Usage: python -m evaluation_function.dev <answer> <response>")
        return
    
    answer = sys.argv[1]
    response = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "basic_nn"
    refresh = sys.argv[4].lower() == "true" if len(sys.argv) >= 4 else False
    params = Params(model=model, refresh=refresh)

    result = evaluation_function(answer, response, params)

    print(result.to_dict())

if __name__ == "__main__":
    dev()