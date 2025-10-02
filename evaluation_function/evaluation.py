from typing import Any
from lf_toolkit.evaluation import Result, Params

from . import models

def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Evaluation Function.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """

    #model_name = getattr(params, "model", "basic_nn")  # default
    model_name = params.get("model", "basic_nn") # default

    print(response, answer, params)
    try:
        model = getattr(models, model_name)   # e.g. models.basic_nn
    except AttributeError:
        raise ValueError(f"Unknown model: {model_name}")

    if not hasattr(model, "run"):
        raise ValueError(f"Model {model_name} has no run()")

    return model.run(response, answer, params)
