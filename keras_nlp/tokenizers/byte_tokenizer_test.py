# Copyright 2022 The KerasNLP Authors
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

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.tokenizers.byte_tokenizer import ByteTokenizer


class ByteTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def test_tokenize(self):
        input_data = tf.constant(["hello", "fun", "▀▁▂▃"])
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        exp_outputs = [
            [104, 101, 108, 108, 111],
            [102, 117, 110],
            [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
        ]
        for i in range(call_output.shape[0]):
            self.assertAllEqual(call_output[i], exp_outputs[i])
            self.assertAllEqual(tokenize_output[i], exp_outputs[i])

    def test_tokenize_scalar(self):
        input_data = "hello"
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 101, 108, 108, 111])
        self.assertAllEqual(tokenize_output, [104, 101, 108, 108, 111])

    def test_dense_output(self):
        input_data = tf.constant(["hello", "fun", "▀▁▂▃"])
        tokenizer = ByteTokenizer(sequence_length=10)
        call_output = tokenizer(input_data)
        self.assertIsInstance(call_output, tf.Tensor)
        self.assertAllEqual(
            call_output,
            [
                [104, 101, 108, 108, 111, 0, 0, 0, 0, 0],
                [102, 117, 110, 0, 0, 0, 0, 0, 0, 0],
                [226, 150, 128, 226, 150, 129, 226, 150, 130, 226],
            ],
        )

    def test_detokenize(self):
        input_data = tf.ragged.constant(
            [
                [104, 101, 108, 108, 111],
                [102, 117, 110],
                [226, 150, 128, 226, 150, 129, 226, 150, 130, 226, 150, 131],
            ]
        )

        tokenizer = ByteTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["hello", "fun", "▀▁▂▃"])

    def test_detokenize_replace_error(self):
        # 226 is an invalid UTF-8 byte.
        input_data = tf.ragged.constant([[104, 101, 226, 150, 108, 108, 111]])

        tokenizer = ByteTokenizer(errors="replace", replacement_char=341)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"he\xc5\x95llo"])

    def test_detokenize_ignore_error(self):
        input_data = tf.ragged.constant([[104, 101, 226, 150, 108, 108, 111]])

        tokenizer = ByteTokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"hello"])

    def test_detokenize_strict_error(self):
        input_data = tf.ragged.constant([[104, 101, 226, 150, 108, 108, 111]])

        tokenizer = ByteTokenizer(errors="strict")
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _ = tokenizer.detokenize(input_data)

    def test_vocab_size(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocabulary_size(), 256)

    def test_lowercase(self):
        input_data = tf.constant(["HeLlO wOrLd"])
        tokenizer = ByteTokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]],
        )

    def test_skip_lowercase(self):
        input_data = tf.constant(["HeLlO wOrLd"])
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
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

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
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

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
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

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
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_functional_model(self):
        input_data = tf.constant(["hello", "fun", "▀▁▂▃"])
        tokenizer = ByteTokenizer()
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(model_output, ["hello", "fun", "▀▁▂▃"])

    def test_load_model_with_config(self):
        input_data = tf.constant(["hello"])

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

        tokenizer = ByteTokenizer(
            name="byte_tokenizer_config_test",
            lowercase=False,
            sequence_length=8,
            normalization_form="NFC",
            errors="ignore",
            replacement_char=0,
        )
        exp_config = {
            "dtype": "int32",
            "errors": "ignore",
            "lowercase": False,
            "name": "byte_tokenizer_config_test",
            "normalization_form": "NFC",
            "replacement_char": 0,
            "sequence_length": 8,
            "trainable": True,
        }
        self.assertEqual(tokenizer.get_config(), exp_config)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["this is fun"])

        tokenizer = ByteTokenizer(
            name="byte_tokenizer_config_test",
            lowercase=False,
            sequence_length=20,
            normalization_form="NFKC",
            errors="replace",
        )
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
