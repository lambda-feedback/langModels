import os
import unittest
from unittest import mock

from .evaluation import Params, evaluation_function

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """

    def test_evaluation(self):
        params = {
            "response": 1,
            "answer": 1,
            "params": {}
        }

        result = evaluation_function(1, 1, params).to_dict()

        self.assertEqual(result.get("is_correct"), True)
        feedback = result.get("feedback", None)
        if isinstance(feedback, list) and feedback:
            self.assertIsInstance(feedback[0][1], str)
        else:
            self.assertIsInstance(feedback, str)


class TestHFModels(unittest.TestCase):
    """GPT-2 and SmolLM2 tests. Generation needs the weights (baked into the image,
    or ~500MB Hugging Face downloads each), so it only runs when RUN_HF_MODEL_TESTS is set."""

    @unittest.skipUnless(os.environ.get("RUN_HF_MODEL_TESTS"), "set RUN_HF_MODEL_TESTS=1 to run")
    def test_gpt2_continues_prompt(self):
        prompt = "Once upon a time"
        result = evaluation_function(
            prompt, None, {"model": "gpt2_infer", "word_count": 5, "seed": 0}
        ).to_dict()
        self.assertTrue(result.get("is_correct"))
        self.assertTrue(result["feedback"].startswith(prompt))
        self.assertGreater(len(result["feedback"]), len(prompt))

    @unittest.skipUnless(os.environ.get("RUN_HF_MODEL_TESTS"), "set RUN_HF_MODEL_TESTS=1 to run")
    def test_smollm2_replies_without_prompt(self):
        prompt = "What is the capital of France?"
        result = evaluation_function(
            prompt, None, {"model": "smollm2_infer", "word_count": 20, "seed": 0}
        ).to_dict()
        self.assertTrue(result.get("is_correct"))
        self.assertTrue(result["feedback"])
        self.assertNotIn(prompt, result["feedback"])
        self.assertNotIn("<|im_", result["feedback"])

    def test_top_k_clamped_to_vocab(self):
        from .models import gpt2_infer, smollm2_infer
        from .models.hf_lm import sampling_params

        for module, vocab_size in [(gpt2_infer, 50257), (smollm2_infer, 49152)]:
            sampling = sampling_params({"top_k": 10**9}, module.DEFAULT_TOKENS, module.MAX_TOKENS, module.VOCAB_SIZE)
            self.assertEqual(sampling["top_k"], vocab_size)

    def test_load_failure_returns_error_result(self):
        from .models import gpt2_infer, smollm2_infer

        for name, module in [("gpt2_infer", gpt2_infer), ("smollm2_infer", smollm2_infer)]:
            with self.subTest(model=name), \
                    mock.patch.object(module.MODEL, "load", side_effect=OSError("no weights")):
                result = evaluation_function("hello", None, {"model": name}).to_dict()
                self.assertFalse(result.get("is_correct"))
                self.assertIn("no weights", str(result.get("feedback")))
