import unittest

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