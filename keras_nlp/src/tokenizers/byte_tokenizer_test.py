# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.tokenizers.byte_tokenizer import ByteTokenizer


class ByteTokenizerTest(TestCase):
    def test_tokenize(self):
        input_data = ["hello", "fun", "▀▁▂▃"]
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        exp_outputs = [
            [104, 101, 108, 108, 111],
            [102, 117, 110],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
        ]
        self.assertAllEqual(call_output, exp_outputs)
        self.assertAllEqual(tokenize_output, exp_outputs)

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
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _ = tokenizer.detokenize(input_data)

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

    def test_tokenize_first_batch_second(self):
        tokenizer = ByteTokenizer()

        ds = tf.data.Dataset.from_tensor_slices(
            ["hello", "fun", "▀▁▂▃", "haha"]
        )
        ds = ds.map(tokenizer)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(4))
        output = ds.take(1).get_single_element()

        exp_output = [
            [104, 101, 108, 108, 111],
            [102, 117, 110],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
            [104, 97, 104, 97],
        ]
        self.assertAllEqual(output, exp_output)

    def test_tokenize_first_batch_second_with_sequence_length(self):
        tokenizer = ByteTokenizer(sequence_length=10)

        ds = tf.data.Dataset.from_tensor_slices(
            ["hello", "fun", "▀▁▂▃", "haha"]
        )
        ds = ds.map(tokenizer)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(4))
        output = ds.take(1).get_single_element()

        exp_output = [
            [104, 101, 108, 108, 111, 0, 0, 0, 0, 0],
            [102, 117, 110, 0, 0, 0, 0, 0, 0, 0],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226],
            [104, 97, 104, 97, 0, 0, 0, 0, 0, 0],
        ]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_tokenize_second(self):
        tokenizer = ByteTokenizer()

        ds = tf.data.Dataset.from_tensor_slices(
            ["hello", "fun", "▀▁▂▃", "haha"]
        )
        ds = ds.batch(4).map(tokenizer)
        output = ds.take(1).get_single_element()

        exp_output = [
            [104, 101, 108, 108, 111],
            [102, 117, 110],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
            [104, 97, 104, 97],
        ]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_tokenize_second_with_sequence_length(self):
        tokenizer = ByteTokenizer(sequence_length=10)

        ds = tf.data.Dataset.from_tensor_slices(
            ["hello", "fun", "▀▁▂▃", "haha"]
        )
        ds = ds.batch(4).map(tokenizer)
        output = ds.take(1).get_single_element()

        exp_output = [
            [104, 101, 108, 108, 111, 0, 0, 0, 0, 0],
            [102, 117, 110, 0, 0, 0, 0, 0, 0, 0],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226],
            [104, 97, 104, 97, 0, 0, 0, 0, 0, 0],
        ]
        self.assertAllEqual(output, exp_output)

    def test_load_model_with_config(self):
        input_data = ["hello"]

        original_tokenizer = ByteTokenizer(
            lowercase=False,
            sequence_length=8,
            normalization_form="NFC",
            errors="ignore",
        )
        cloned_tokenizer = ByteTokenizer.from_config(
            original_tokenizer.get_config()
        )
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

        decoded_input = [[104, 101, 226, 150, 108, 108, 111]]
        self.assertAllEqual(
            original_tokenizer.detokenize(decoded_input),
            cloned_tokenizer.detokenize(decoded_input),
        )

    def test_config(self):
        input_data = ["hello", "fun", "▀▁▂▃", "haha"]
        tokenizer = ByteTokenizer(
            name="byte_tokenizer_config_test",
            lowercase=False,
            sequence_length=8,
            normalization_form="NFC",
            errors="ignore",
            replacement_char=0,
        )
        cloned_tokenizer = ByteTokenizer.from_config(tokenizer.get_config())
        self.assertAllEqual(
            tokenizer(input_data),
            cloned_tokenizer(input_data),
        )
