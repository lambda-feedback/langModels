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


class TestGPT2(unittest.TestCase):
    """GPT-2 tests. The happy path needs the weights (baked into the image, or a
    ~500MB Hugging Face download), so it only runs when RUN_GPT2_TESTS is set."""

    @unittest.skipUnless(os.environ.get("RUN_GPT2_TESTS"), "set RUN_GPT2_TESTS=1 to run")
    def test_generates_continuation(self):
        prompt = "Once upon a time"
        result = evaluation_function(
            prompt, None, {"model": "gpt2_infer", "word_count": 5, "seed": 0}
        ).to_dict()
        self.assertTrue(result.get("is_correct"))
        self.assertTrue(result["feedback"].startswith(prompt))
        self.assertGreater(len(result["feedback"]), len(prompt))

    def test_load_failure_returns_error_result(self):
        from .models import gpt2_infer

        with mock.patch.object(gpt2_infer, "_load", side_effect=OSError("no weights")):
            result = evaluation_function(
                "hello", None, {"model": "gpt2_infer"}
            ).to_dict()
        self.assertFalse(result.get("is_correct"))
        self.assertIn("no weights", str(result.get("feedback")))