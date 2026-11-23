from unittest import mock

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
            [
                "<img>",
                "the quick brown fox",
                "<img> earth is round \n\n <img>",
                "the    fox",
                " the fox ",
                "\xa0the fox\xa0",
                "\u2003the fox\u2003",
                "\u3000the fox\u3000",
                "<img>the<end_of_image><img>",
            ],
        )

    def test_gemma3n_parity(self):
        self._test_tokenizer_parity(
            MockGemma3nTokenizer,
            "<audio_soft_token> the quick brown fox <img>",
            [
                "<audio_soft_token>",
                "the quick brown fox",
                "<img> earth is round \n\n <audio_soft_token>",
                "the    fox",
                " the fox ",
                "\xa0the fox\xa0",
                "\u2003the fox\u2003",
                "\u3000the fox\u3000",
                "<audio_soft_token>the<end_of_audio><audio_soft_token>",
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
                "the    fox",
                " the fox ",
                "\xa0the fox\xa0",
                "\u2003the fox\u2003",
                "\u3000the fox\u3000",
                "<|video|>the<|audio|>",
                "<|image>the<image|>",
                "<|video>the<video|>",
                "<|audio>the<audio|>",
            ],
        )

    def test_leading_unk_quirk(self):
        # Pins the boundary `<unk>` quirk for Gemma3/Gemma3n.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        tokenizer = MockGemma3Tokenizer()

        # Space-padded special tokens at a string boundary leave an empty
        # field that misses the vocabulary and becomes `<unk>` (3).
        tf_out = ops.convert_to_numpy(tokenizer._tokenize_tf("<img>")).tolist()
        py_out = tokenizer._tokenize_python("<img>").tolist()
        self.assertEqual(tf_out, [3, tokenizer.token_to_id("<img>"), 3])
        self.assertEqual(py_out, [3, tokenizer.token_to_id("<img>"), 3])

        # Mid-string, the padding is absorbed by the surrounding spaces.
        mid = tokenizer._tokenize_python("the <img> fox").tolist()
        self.assertNotIn(3, mid)

        # Leading whitespace alone is enough to produce the `<unk>`.
        lead = tokenizer._tokenize_python(" the fox").tolist()
        self.assertEqual(lead[0], 3)

    def test_public_dispatch_uses_python_workflow(self):
        # Eager calls must never hit the TF path; concrete values also pin
        # `add_bos`, `add_eos` and `detokenize` stripping when `tf is None`.
        cases = [
            (MockGemma3Tokenizer, "the <img> fox", [9, 8, 12]),
            (MockGemma3nTokenizer, "the <audio_soft_token> fox", [9, 17, 12]),
            (MockGemma4Tokenizer, "the <|video|> fox", [9, 22, 12]),
        ]
        for tokenizer_cls, text, expected_ids in cases:
            boom = AssertionError("TF path used outside a tf.function")
            with (
                mock.patch.object(
                    tokenizer_cls, "_tokenize_tf", side_effect=boom
                ),
                mock.patch.object(
                    tokenizer_cls, "_detokenize_tf", side_effect=boom
                ),
            ):
                tokenizer = tokenizer_cls()
                self.assertEqual(
                    tokenizer.tokenize(text).tolist(), expected_ids
                )
                self.assertEqual(tokenizer.detokenize(expected_ids), text)

                tokenizer = tokenizer_cls(add_bos=True, add_eos=True)
                with_specials = [1] + expected_ids + [2]
                self.assertEqual(
                    tokenizer.tokenize(text).tolist(), with_specials
                )
                self.assertEqual(
                    tokenizer(text, training=False).tolist(), with_specials
                )
                self.assertEqual(
                    tokenizer.detokenize(with_specials + [0]), text
                )

    def test_tf_data_map_retraces(self):
        # Pins `tf.init_scope()` in `_maybe_initialized_tf`: without it the
        # lookup tables are captured by the first trace and the second fails.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        tokenizer = MockGemma3Tokenizer()
        for _ in range(2):
            ds = tf.data.Dataset.from_tensor_slices(["the <img> fox"])
            ds = ds.map(tokenizer.tokenize)
            self.assertEqual(next(iter(ds)).numpy().tolist(), [9, 8, 12])

    def test_python_tokenize_dtype_string(self):
        for cls in (
            MockGemma3Tokenizer,
            MockGemma3nTokenizer,
            MockGemma4Tokenizer,
        ):
            tokenizer = cls(dtype="string")
            tokens = tokenizer.tokenize("the fox")
            self.assertEqual(tokens, ["the", "fox"])
            self.assertIsInstance(tokens, list)
            self.assertEqual(
                tokenizer("the fox", training=False), ["the", "fox"]
            )
            self.assertEqual(
                tokenizer.tokenize(["the fox", "unknown"]),
                [["the", "fox"], ["<unk>"]],
            )

            tokenizer = cls(dtype="string", add_bos=True, add_eos=True)
            expected = ["<bos>", "the", "fox", "<eos>"]
            self.assertEqual(tokenizer.tokenize("the fox"), expected)
            if tf is not None:
                # TF string outputs arrive as decoded Python lists.
                self.assertEqual(tokenizer._tokenize_tf("the fox"), expected)

    def test_tf_is_none_raises_clear_error(self):
        for cls in (
            MockGemma3Tokenizer,
            MockGemma3nTokenizer,
            MockGemma4Tokenizer,
        ):
            with mock.patch("keras_hub.src.utils.tensor_utils.tf", None):
                tokenizer = cls()
                with self.assertRaisesRegex(
                    ImportError, rf"{cls.__name__} requires `tensorflow`"
                ):
                    cls._tokenize_tf.__wrapped__(tokenizer, ["the fox"])
                with self.assertRaisesRegex(
                    ImportError, rf"{cls.__name__} requires `tensorflow`"
                ):
                    cls._detokenize_tf.__wrapped__(tokenizer, [9, 12])

    def test_get_config_round_trip(self):
        for cls in (
            MockGemma3Tokenizer,
            MockGemma3nTokenizer,
            MockGemma4Tokenizer,
        ):
            tokenizer = cls(
                sequence_length=12, dtype="int64", add_bos=True, add_eos=True
            )
            reloaded = cls.from_config(tokenizer.get_config())
            self.assertEqual(reloaded.sequence_length, 12)
            self.assertEqual(reloaded.compute_dtype, "int64")
            self.assertTrue(reloaded.add_bos)
            self.assertTrue(reloaded.add_eos)
            tokens = reloaded.tokenize("the fox")
            self.assertEqual(str(tokens.dtype), "int64")
            self.assertEqual(tokens.tolist(), [1, 9, 12, 2])
            self.assertEqual(reloaded.detokenize(tokens), "the fox")

    def _test_tokenizer_parity(
        self, tokenizer_cls, scalar_input, batched_input
    ):
        # Nothing to compare against when there is no TF path to run.
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        for dtype in ["int32", "int64", "string"]:
            for add_bos in [True, False]:
                for add_eos in [True, False]:
                    tokenizer = tokenizer_cls(
                        add_bos=add_bos, add_eos=add_eos, dtype=dtype
                    )
                    # Unbatched int outputs: backend tensor (TF) / ndarray
                    # (Python). String outputs are lists on both paths.
                    tf_scalar = tokenizer._tokenize_tf(scalar_input)
                    py_scalar = tokenizer._tokenize_python(scalar_input)
                    if dtype != "string":
                        self.assertEqual(str(py_scalar.dtype), dtype)
                        # JAX truncates int64 to int32 in `convert_to_tensor`
                        # without x64, so compare against the same conversion.
                        tf_scalar = ops.convert_to_numpy(tf_scalar)
                        self.assertEqual(
                            tf_scalar.dtype,
                            ops.convert_to_numpy(
                                ops.convert_to_tensor(py_scalar)
                            ).dtype,
                        )
                        tf_scalar = tf_scalar.tolist()
                        py_scalar = py_scalar.tolist()
                    self.assertEqual(tf_scalar, py_scalar)

                    # Pin `_cast_tokens_tf` on the raw tensor `tf.data` sees.
                    raw = tokenizer._tokenize_tf.__wrapped__(
                        tokenizer, scalar_input
                    )
                    self.assertEqual(raw.dtype, tf.as_dtype(dtype))

                    # Batched outputs are ragged, so lists on both paths.
                    tf_batch = tokenizer._tokenize_tf(batched_input)
                    py_batch = tokenizer._tokenize_python(batched_input)
                    self.assertEqual(tf_batch, py_batch)

                    if dtype == "string":
                        continue
                    self.assertEqual(
                        tokenizer._detokenize_tf(tf_scalar),
                        tokenizer._detokenize_python(py_scalar),
                    )
                    self.assertEqual(
                        tokenizer._detokenize_tf(tf_batch),
                        tokenizer._detokenize_python(py_batch),
                    )
