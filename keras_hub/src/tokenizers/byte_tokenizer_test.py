import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.byte_tokenizer import ByteTokenizer


class ByteTokenizerTest(TestCase):
    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ByteTokenizer,
            init_kwargs={},
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
            init_kwargs={"sequence_length": 12},
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
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 101, 108, 108, 111])
        self.assertAllEqual(tokenize_output, [104, 101, 108, 108, 111])

    def test_dense_output(self):
        input_data = ["hello", "fun", "▀▁▂▃"]
        tokenizer = ByteTokenizer(sequence_length=10)
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

        tokenizer = ByteTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["hello", "fun", "▀▁▂▃"])

    def test_detokenize_replace_error(self):
        # 226 is an invalid UTF-8 byte.
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = ByteTokenizer(errors="replace", replacement_char=341)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"he\xc5\x95llo"])

    def test_detokenize_ignore_error(self):
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = ByteTokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"hello"])

    def test_detokenize_strict_error(self):
        input_data = [[104, 101, 226, 150, 108, 108, 111]]

        tokenizer = ByteTokenizer(errors="strict")
        expected_errors = (ValueError,)
        if tf is not None:
            expected_errors = (ValueError, tf.errors.InvalidArgumentError)
        with self.assertRaises(expected_errors):
            _ = tokenizer.detokenize(input_data)

    def test_detokenize_replace_valid_chars(self):
        # 255 is invalid, 239,191,189 is valid U+FFFD.
        # The invalid byte should be replaced by 'H' (72), but the valid
        # U+FFFD should remain.
        input_data = [[104, 101, 255, 108, 108, 111, 239, 191, 189]]
        tokenizer = ByteTokenizer(errors="replace", replacement_char=72)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["heHllo\ufffd"])

    def test_workflow_parity(self):
        if tf is None:
            return  # Skip if TensorFlow is not available

        input_data = ["hello", "fun", "▀▁▂▃", "haha"]
        tokenizer = ByteTokenizer(sequence_length=12)

        # Force TF Workflow
        tokenizer._allow_python_workflow = False
        tf_out = tokenizer(input_data)
        tf_detok = tokenizer.detokenize(tf_out)

        # Force Python Workflow
        tokenizer._allow_python_workflow = True
        python_out = tokenizer(input_data)
        python_detok = tokenizer.detokenize(python_out)

        self.assertAllEqual(tf_out, python_out)
        self.assertAllEqual(tf_detok, python_detok)

    def test_vocab_size(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocabulary_size(), 256)

    def test_lowercase(self):
        input_data = ["HeLlO wOrLd"]
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]],
        )

    def test_skip_lowercase(self):
        input_data = ["HeLlO wOrLd"]
        tokenizer = ByteTokenizer(lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output, [[72, 101, 76, 108, 79, 32, 119, 79, 114, 76, 100]]
        )

    def test_token_to_id(self):
        input_tokens = ["f", "u", "n"]
        expected_ids = [102, 117, 110]
        tokenizer = ByteTokenizer()
        ids = [tokenizer.token_to_id(t) for t in input_tokens]
        self.assertAllEqual(ids, expected_ids)

    def test_id_to_token(self):
        input_ids = [102, 117, 110]
        expected_tokens = ["f", "u", "n"]
        tokenizer = ByteTokenizer()
        tokens = [tokenizer.id_to_token(i) for i in input_ids]
        self.assertAllEqual(tokens, expected_tokens)
