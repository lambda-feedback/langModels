"""Model registry.

Model modules are imported lazily by ``evaluation_function.evaluation`` (not
here) so that a cold start does not pay torch's multi-second import cost until a
request actually selects a model that needs it.
"""

AVAILABLE_MODELS = [
    "basic_nn",
    "shannon_letters_single",
    "shannon_letters_ngram",
    "shannon_words_ngram",
    "bengio_infer",
]

__all__ = ["AVAILABLE_MODELS"]