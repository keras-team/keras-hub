import numpy as np
import pytest
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.byte_tokenizer import ByteTokenizer

try:
    import grain
except ImportError:
    grain = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


class ByteTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self._allow_python_workflow = True

    def make_tokenizer(self, **kwargs):
        return ByteTokenizer(
            _allow_python_workflow=self._allow_python_workflow, **kwargs
        )

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ByteTokenizer,
            init_kwargs={
                "_allow_python_workflow": self._allow_python_workflow,
            },
            input_data=["hello", "fun", "▀▁▂▃", "haha"],
            expected_output=[
                [104, 101, 108, 108, 111],
                [102, 117, 110],
                [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
                [104, 97, 104, 97],
            ],
        )

    def test_tokenizer_basics_with_sequence_length(self):
        # `sequence_length=12` is long enough that none of the inputs below
        # get truncated, so the dense, padded output round-trips cleanly
        # through `detokenize`.
        self.run_preprocessing_layer_test(
            cls=ByteTokenizer,
            init_kwargs={
                "sequence_length": 12,
                "_allow_python_workflow": self._allow_python_workflow,
            },
            input_data=["hello", "fun", "▀▁▂▃", "haha"],
            expected_output=[
                [104, 101, 108, 108, 111, 0, 0, 0, 0, 0, 0, 0],
                [102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
                [104, 97, 104, 97, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_tokenize_scalar(self):
        input_data = "hello"
        tokenizer = self.make_tokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 101, 108, 108, 111])
        self.assertAllEqual(tokenize_output, [104, 101, 108, 108, 111])

    def test_dense_output(self):
        input_data = ["hello", "fun", "▀▁▂▃"]
        tokenizer = self.make_tokenizer(sequence_length=10)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [
                [104, 101, 108, 108, 111, 0, 0, 0, 0, 0],
                [102, 117, 110, 0, 0, 0, 0, 0, 0, 0],
                [226, 150, 128, 226, 150, 129, 226, 150, 130, 226],
            ],
        )

    def test_detokenize(self):
        input_data = [
            [104, 101, 108, 108, 111],
            [102, 117, 110],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
        ]

        tokenizer = self.make_tokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["hello", "fun", "▀▁▂▃"])

    def test_detokenize_replace_error(self):
        # 226 is an invalid UTF-8 byte.
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = self.make_tokenizer(errors="replace", replacement_char=341)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"he\xc5\x95llo"])

    def test_detokenize_ignore_error(self):
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = self.make_tokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"hello"])

    def test_detokenize_strict_error(self):
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = self.make_tokenizer(errors="strict")
        if self._allow_python_workflow:
            error = ValueError
        else:
            error = tf.errors.InvalidArgumentError
        with self.assertRaises(error):
            _ = tokenizer.detokenize(input_data)

    def test_vocab_size(self):
        tokenizer = self.make_tokenizer()
        self.assertEqual(tokenizer.vocabulary_size(), 256)

    def test_lowercase(self):
        input_data = ["HeLlO wOrLd"]
        tokenizer = self.make_tokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]],
        )

    def test_skip_lowercase(self):
        input_data = ["HeLlO wOrLd"]
        tokenizer = self.make_tokenizer(lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output, [[72, 101, 76, 108, 79, 32, 119, 79, 114, 76, 100]]
        )

    def test_token_to_id(self):
        input_tokens = ["f", "u", "n"]
        expected_ids = [102, 117, 110]
        tokenizer = self.make_tokenizer()
        ids = [tokenizer.token_to_id(t) for t in input_tokens]
        self.assertAllEqual(ids, expected_ids)

    def test_id_to_token(self):
        input_ids = [102, 117, 110]
        expected_tokens = ["f", "u", "n"]
        tokenizer = self.make_tokenizer()
        tokens = [tokenizer.id_to_token(i) for i in input_ids]
        self.assertAllEqual(tokens, expected_tokens)

    def test_bytes_and_tensor_inputs(self):
        if not self._allow_python_workflow:
            self.skipTest("The TF path does not accept numpy string arrays.")
        tokenizer = self.make_tokenizer()
        expected = [[104, 101, 108, 108, 111], [102, 117, 110]]
        self.assertAllEqual(tokenizer([b"hello", b"fun"]), expected)
        self.assertAllEqual(tokenizer(np.array(["hello", "fun"])), expected)
        self.assertAllEqual(tokenizer(tf.constant(["hello", "fun"])), expected)
        self.assertAllEqual(tokenizer(b"hello"), expected[0])
        self.assertAllEqual(tokenizer(tf.constant("hello")), expected[0])
        # Invalid utf-8 bytes are passed through.
        self.assertAllEqual(tokenizer(b"h\xffi"), [104, 255, 105])

    def test_detokenize_inputs(self):
        if not self._allow_python_workflow:
            self.skipTest("The TF path does not accept int64 inputs.")
        tokenizer = self.make_tokenizer()
        ids = [[104, 101, 108, 108, 111, 0, 0], [102, 117, 110, 0, 0, 0, 0]]
        self.assertAllEqual(tokenizer.detokenize(ids), ["hello", "fun"])
        self.assertAllEqual(
            tokenizer.detokenize(np.array(ids)), ["hello", "fun"]
        )
        self.assertAllEqual(
            tokenizer.detokenize(tf.constant(ids)), ["hello", "fun"]
        )
        self.assertAllEqual(tokenizer.detokenize(ids[0]), "hello")
        self.assertAllEqual(tokenizer.detokenize(np.array(ids[0])), "hello")

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_outputs_numpy(self):
        input_data = ["hello", "fun"]
        # Ragged outputs are python lists.
        tokenizer = self.make_tokenizer()
        (outputs,) = list(grain.MapDataset.source([input_data]).map(tokenizer))
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, [[104, 101, 108, 108, 111], [102, 117, 110]])
        # Dense outputs are numpy arrays.
        tokenizer = self.make_tokenizer(sequence_length=6)
        outputs = list(grain.MapDataset.source(input_data).map(tokenizer))
        for output in outputs:
            self.assertIsInstance(output, np.ndarray)
        (batch,) = list(
            grain.MapDataset.source(input_data).map(tokenizer).batch(2)
        )
        self.assertAllEqual(
            batch, [[104, 101, 108, 108, 111, 0], [102, 117, 110, 0, 0, 0]]
        )
        self.assertAllEqual(batch, tokenizer(input_data))

    @pytest.mark.skipif(tf_text is None, reason="tensorflow-text not installed")
    def test_python_path_matches_tf_path(self):
        if not self._allow_python_workflow:
            self.skipTest("Parity test only runs from the python path.")
        corpus = [
            "The quick brown fox.",
            "ﬁsh ǅ ß İ Σ",
            "á é í ó ú Ç",
            "  leading and trailing  ",
            "x\xa0y\u3000z\ty\nq",
            "emoji 😀 test 한국어 テスト",
            "",
        ]
        bad_ids = [
            [104, 101, 226, 150, 108, 108, 111],
            [255],
            [226, 150, 128, 226],
            [237, 160, 128],
        ]
        for kwargs in [
            {},
            {"lowercase": False},
            {"normalization_form": "NFKD"},
            {"lowercase": False, "normalization_form": "NFC"},
            {"sequence_length": 8},
            {"errors": "replace", "replacement_char": 88},
            {"errors": "ignore"},
        ]:
            python_tokenizer = ByteTokenizer(**kwargs)
            tf_tokenizer = ByteTokenizer(_allow_python_workflow=False, **kwargs)
            python_ids = python_tokenizer(corpus)
            tf_ids = tf_tokenizer(corpus)
            self.assertAllEqual(python_ids, tf_ids)
            self.assertAllEqual(
                python_tokenizer.detokenize(python_ids),
                tf_tokenizer.detokenize(tf_ids),
            )
            self.assertAllEqual(
                python_tokenizer.detokenize(bad_ids),
                tf_tokenizer.detokenize(bad_ids),
            )


class ByteTokenizerTFTest(ByteTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        super().setUp()
        self._allow_python_workflow = False
