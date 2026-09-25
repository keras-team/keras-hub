from unittest import mock

import numpy as np
from keras import ops

from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import MockGemma3Tokenizer
from keras_hub.src.tests.mocks.mock_gemma3n_tokenizer import (
    MockGemma3nTokenizer,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class MockGemmaTokenizersTest(TestCase):
    def test_gemma3_parity(self):
        self._test_tokenizer_parity(
            MockGemma3Tokenizer,
            "<img> the quick brown fox",
            ["<img>", "the quick brown fox", "<img> earth is round \n\n <img>"],
        )

    def test_gemma3n_parity(self):
        self._test_tokenizer_parity(
            MockGemma3nTokenizer,
            "<audio_soft_token> the quick brown fox <img>",
            [
                "<audio_soft_token>",
                "the quick brown fox",
                "<img> earth is round \n\n <audio_soft_token>",
            ],
        )

    def test_gemma4_parity(self):
        self._test_tokenizer_parity(
            MockGemma4Tokenizer,
            "<|video|> the quick brown fox <|audio|>",
            [
                "<|video|>",
                "the quick brown fox",
                "<|image|> earth is round \n\n <|video|>",
            ],
        )

    def test_leading_unk_quirk(self):
        # Pins the boundary `<unk>` quirk for Gemma3/Gemma3n.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        tokenizer = MockGemma3Tokenizer()

        # These tokenizers pad special tokens with spaces and never strip, so
        # a special token at the start or end of the string leaves an empty
        # field on the outside of it. Splitting on `sep=" "` keeps those empty
        # fields, and they miss in the vocabulary, so they take `default_value`
        # 3 (`<unk>`). It is the string boundary that does this, not the
        # special token -- see the two cases below.
        # `_tokenize_tf` is wrapped in `@preprocessing_function`, so its output
        # is a backend tensor. Use `ops.convert_to_numpy` (not `np.array`) so
        # this works when torch places it on a non-CPU device.
        tf_out = ops.convert_to_numpy(tokenizer._tokenize_tf("<img>")).tolist()
        py_out = tokenizer._tokenize_python("<img>")
        if isinstance(py_out, np.ndarray):
            py_out = py_out.tolist()

        # "<img>" becomes " <img> ", which splits to ["", "<img>", ""] and
        # maps to [<unk>, <img>, <unk>].
        self.assertEqual(tf_out, [3, tokenizer.token_to_id("<img>"), 3])
        self.assertEqual(py_out, [3, tokenizer.token_to_id("<img>"), 3])

        # A special token in the middle of the string produces no `<unk>`,
        # because the spaces it is padded with are absorbed by the spaces
        # already separating the words.
        mid = tokenizer._tokenize_python("the <img> fox")
        if isinstance(mid, np.ndarray):
            mid = mid.tolist()
        self.assertNotIn(3, mid)

        # Conversely, leading whitespace alone is enough, with no special
        # token involved at all.
        lead = tokenizer._tokenize_python(" the fox")
        if isinstance(lead, np.ndarray):
            lead = lead.tolist()
        self.assertEqual(lead[0], 3)

    def test_public_dispatch_uses_python_workflow(self):
        # Outside a `tf.function` the public entry points must not touch the
        # TF path. The two paths are value-identical by construction, so
        # comparing outputs cannot detect a broken dispatcher -- make the TF
        # privates explode instead.
        cases = [
            (MockGemma3Tokenizer, "<img> the quick brown fox"),
            (MockGemma3nTokenizer, "<audio_soft_token> the quick brown fox"),
            (MockGemma4Tokenizer, "<|video|> the quick brown fox"),
        ]
        for tokenizer_cls, text in cases:
            tokenizer = tokenizer_cls()
            boom = AssertionError("TF path used outside a tf.function")
            with (
                mock.patch.object(
                    tokenizer_cls, "_tokenize_tf", side_effect=boom
                ),
                mock.patch.object(
                    tokenizer_cls, "_detokenize_tf", side_effect=boom
                ),
            ):
                tokens = tokenizer.tokenize(text)
                text_out = tokenizer.detokenize(tokens)
            self.assertAllEqual(tokens, tokenizer._tokenize_python(text))
            self.assertEqual(text_out, tokenizer._detokenize_python(tokens))

    def test_tf_data_map_retraces(self):
        # Pins the `tf.init_scope()` in `_maybe_initialized_tf`: without it
        # the lookup tables are captured in the first trace's FuncGraph and
        # the second trace fails.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        tokenizer = MockGemma3Tokenizer()
        for _ in range(2):
            ds = tf.data.Dataset.from_tensor_slices(["<img> the fox"])
            ds = ds.map(tokenizer.tokenize)
            self.assertAllEqual(
                next(iter(ds)).numpy().tolist(),
                tokenizer._tokenize_python("<img> the fox").tolist(),
            )

    def _test_tokenizer_parity(
        self, tokenizer_cls, scalar_input, batched_input
    ):
        # Nothing to compare against when there is no TF path to run.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        for add_bos in [True, False]:
            for add_eos in [True, False]:
                tokenizer = tokenizer_cls(add_bos=add_bos, add_eos=add_eos)

                # Test scalar
                # Backend tensor out of `@preprocessing_function`; see the note
                # in `test_leading_unk_quirk`.
                tf_scalar_tokens = ops.convert_to_numpy(
                    tokenizer._tokenize_tf(scalar_input)
                )
                py_scalar_tokens = tokenizer._tokenize_python(scalar_input)
                self.assertAllEqual(tf_scalar_tokens, py_scalar_tokens)

                tf_scalar_detokenized = tokenizer._detokenize_tf(
                    tf_scalar_tokens
                )
                if hasattr(tf_scalar_detokenized, "numpy"):
                    tf_scalar_detokenized = tf_scalar_detokenized.numpy()
                elif hasattr(tf_scalar_detokenized, "item"):
                    tf_scalar_detokenized = tf_scalar_detokenized.item()
                if isinstance(tf_scalar_detokenized, bytes):
                    tf_scalar_detokenized = tf_scalar_detokenized.decode(
                        "utf-8"
                    )

                py_scalar_detokenized = tokenizer._detokenize_python(
                    py_scalar_tokens
                )
                self.assertEqual(tf_scalar_detokenized, py_scalar_detokenized)

                # Test batch
                tf_batch_tokens = tokenizer._tokenize_tf(batched_input)
                py_batch_tokens = tokenizer._tokenize_python(batched_input)

                if hasattr(tf_batch_tokens, "to_list"):
                    tf_batch_tokens_list = tf_batch_tokens.to_list()
                elif hasattr(tf_batch_tokens, "numpy"):
                    tf_batch_tokens_list = tf_batch_tokens.numpy().tolist()
                else:
                    tf_batch_tokens_list = tf_batch_tokens

                # For batched_input python outputs a list of lists.
                if isinstance(py_batch_tokens, np.ndarray):
                    py_batch_tokens_list = py_batch_tokens.tolist()
                else:
                    py_batch_tokens_list = py_batch_tokens

                # Note: assertAllEqual might fail on ragged if omit flat element
                # but comparing lists directly is safe
                self.assertEqual(tf_batch_tokens_list, py_batch_tokens_list)

                tf_batch_detokenized = tokenizer._detokenize_tf(tf_batch_tokens)
                if hasattr(tf_batch_detokenized, "numpy"):
                    tf_batch_detokenized_list = [
                        x.decode("utf-8") if isinstance(x, bytes) else x
                        for x in tf_batch_detokenized.numpy()
                    ]
                elif isinstance(tf_batch_detokenized, list):
                    tf_batch_detokenized_list = [
                        x.decode("utf-8") if isinstance(x, bytes) else x
                        for x in tf_batch_detokenized
                    ]
                else:
                    tf_batch_detokenized_list = [
                        x.decode("utf-8") if isinstance(x, bytes) else x
                        for x in np.array(tf_batch_detokenized).tolist()
                    ]

                py_batch_detokenized_list = tokenizer._detokenize_python(
                    py_batch_tokens
                )
                self.assertEqual(
                    tf_batch_detokenized_list, py_batch_detokenized_list
                )
