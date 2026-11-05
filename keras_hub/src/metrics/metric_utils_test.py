import os
import subprocess
import sys

import numpy as np
import pytest

from keras_hub.src.metrics.metric_utils import rank_of
from keras_hub.src.metrics.metric_utils import to_nested_list
from keras_hub.src.metrics.metric_utils import validate_and_fix_rank
from keras_hub.src.tests.test_case import TestCase

try:
    import tensorflow as tf
except ImportError:
    tf = None


class ToNestedListTest(TestCase):
    def test_python_inputs(self):
        self.assertEqual(to_nested_list("a"), "a")
        self.assertEqual(to_nested_list(b"a"), "a")
        self.assertEqual(to_nested_list(["a", "b"]), ["a", "b"])
        self.assertEqual(to_nested_list(("a", ("b",))), ["a", ["b"]])
        self.assertEqual(
            to_nested_list([["a", "b"], ["c"]]), [["a", "b"], ["c"]]
        )

    def test_numpy_inputs(self):
        self.assertEqual(to_nested_list(np.array("a")), "a")
        self.assertEqual(to_nested_list(np.array([b"a", b"b"])), ["a", "b"])
        self.assertEqual(
            to_nested_list(np.array([["a"], ["b"]])), [["a"], ["b"]]
        )

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_inputs(self):
        self.assertEqual(to_nested_list(tf.constant("a")), "a")
        self.assertEqual(
            to_nested_list(tf.constant([["a"], ["b"]])), [["a"], ["b"]]
        )
        self.assertEqual(
            to_nested_list(tf.ragged.constant([["a", "b"], ["c"]])),
            [["a", "b"], ["c"]],
        )

    def test_rank_of(self):
        self.assertEqual(rank_of("a"), 0)
        self.assertEqual(rank_of(["a"]), 1)
        self.assertEqual(rank_of([["a"], ["b"]]), 2)
        self.assertEqual(rank_of([[["a"]]]), 3)
        self.assertEqual(rank_of([]), 1)


class ValidateAndFixRankTest(TestCase):
    def test_base_rank_zero(self):
        self.assertEqual(validate_and_fix_rank("a", "x"), ["a"])
        self.assertEqual(validate_and_fix_rank(["a", "b"], "x"), ["a", "b"])
        self.assertEqual(validate_and_fix_rank([["a"], ["b"]], "x"), ["a", "b"])

    def test_base_rank_one(self):
        self.assertEqual(
            validate_and_fix_rank(["a", "b"], "x", base_rank=1), [["a", "b"]]
        )
        self.assertEqual(
            validate_and_fix_rank([["a"], ["b", "c"]], "x", base_rank=1),
            [["a"], ["b", "c"]],
        )
        self.assertEqual(
            validate_and_fix_rank([[["a"]], [["b"]]], "x", base_rank=1),
            [["a"], ["b"]],
        )

    def test_invalid_ranks(self):
        with self.assertRaisesRegex(ValueError, "x must be of rank 0, 1, or 2"):
            validate_and_fix_rank([[["a"]]], "x")
        with self.assertRaisesRegex(ValueError, "x must be of rank 1, 2, or 3"):
            validate_and_fix_rank("a", "x", base_rank=1)
        with self.assertRaisesRegex(ValueError, "x is of rank 2"):
            validate_and_fix_rank([["a", "b"]], "x")
        with self.assertRaisesRegex(ValueError, "x is of rank 3"):
            validate_and_fix_rank([[["a", "b"]]], "x", base_rank=1)


class MetricsWithoutTensorFlowTest(TestCase):
    def test_text_metrics_run_without_tensorflow(self):
        # Make `import tensorflow` raise `ImportError` in a fresh interpreter
        # and check that the text metrics still import and compute. This is
        # the regression guard for the TF-free migration of these metrics.
        if not sys.executable:
            self.skipTest("sys.executable is not available.")
        script = """
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_text"] = None
from keras_hub.src.metrics.bleu import Bleu
from keras_hub.src.metrics.rouge_l import RougeL
from keras_hub.src.metrics.rouge_n import RougeN
bleu = Bleu()(
    [
        ["He eats a sweet apple."],
        ["Silicon Valley is one of my favourite shows!"],
    ],
    [
        "He He He eats sweet apple which is a fruit.",
        "I love Silicon Valley, it's one of my favourite shows.",
    ],
)
rouge_n = RougeN(order=2)("the cat was under the bed", "the cat was found")
rouge_l = RougeL()("hello this is fun", "hello this is awesome")
assert sys.modules["tensorflow"] is None
print(f"{float(bleu):.3f} {float(rouge_n['f1_score']):.3f} "
      f"{float(rouge_l['f1_score']):.3f}")
"""
        # The NumPy backend needs no framework, so the subprocess can run
        # whatever backend the test session itself is using.
        env = dict(os.environ, KERAS_BACKEND="numpy")
        output = subprocess.check_output(
            [sys.executable, "-c", script], env=env, stderr=subprocess.DEVNULL
        )
        self.assertEqual(output.decode("utf-8").strip(), "0.243 0.500 0.750")
