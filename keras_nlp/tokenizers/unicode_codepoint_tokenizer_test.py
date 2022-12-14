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

from keras_nlp.tokenizers.unicode_codepoint_tokenizer import (
    UnicodeCodepointTokenizer,
)


class UnicodeCodepointTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def test_tokenize(self):
        input_data = tf.constant(["ninja", "samurai", "▀▁▂▃"])
        tokenizer = UnicodeCodepointTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        exp_outputs = [
            [110, 105, 110, 106, 97],
            [115, 97, 109, 117, 114, 97, 105],
            [9600, 9601, 9602, 9603],
        ]
        for i in range(call_output.shape[0]):
            self.assertAllEqual(call_output[i], exp_outputs[i])
            self.assertAllEqual(tokenize_output[i], exp_outputs[i])

    def test_tokenize_scalar(self):
        input_data = "ninja"
        tokenizer = UnicodeCodepointTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [110, 105, 110, 106, 97])
        self.assertAllEqual(tokenize_output, [110, 105, 110, 106, 97])

    def test_dense_output(self):
        input_data = tf.constant(["ninja", "samurai", "▀▁▂▃"])
        tokenizer = UnicodeCodepointTokenizer(sequence_length=10)
        call_output = tokenizer(input_data)
        self.assertIsInstance(call_output, tf.Tensor)
        self.assertAllEqual(
            call_output,
            [
                [110, 105, 110, 106, 97, 0, 0, 0, 0, 0],
                [115, 97, 109, 117, 114, 97, 105, 0, 0, 0],
                [9600, 9601, 9602, 9603, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_tokenize_scalar_with_vocabulary_size(self):
        input_data = "ninja"
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=105)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 104, 104, 104, 97])
        self.assertAllEqual(tokenize_output, [104, 104, 104, 104, 97])

    def test_tokenize_dense_with_vocabulary_size(self):
        input_data = tf.constant(["ninja", "samurai", "▀▁▂▃"])
        tokenizer = UnicodeCodepointTokenizer(
            sequence_length=10, vocabulary_size=105
        )
        call_output = tokenizer(input_data)
        self.assertIsInstance(call_output, tf.Tensor)
        self.assertAllEqual(
            call_output,
            [
                [104, 104, 104, 104, 97, 0, 0, 0, 0, 0],
                [104, 97, 104, 104, 104, 97, 104, 0, 0, 0],
                [104, 104, 104, 104, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_tokenize_ragged_with_vocabulary_size(self):
        input_data = tf.constant(["ninja", "samurai", "▀▁▂▃"])
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=105)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        exp_outputs = [
            [104, 104, 104, 104, 97],
            [104, 97, 104, 104, 104, 97, 104],
            [104, 104, 104, 104],
        ]
        for i in range(call_output.shape[0]):
            self.assertAllEqual(call_output[i], exp_outputs[i])
            self.assertAllEqual(tokenize_output[i], exp_outputs[i])

    def test_detokenize(self):
        input_data = tf.ragged.constant(
            [
                [110, 105, 110, 106, 97],
                [115, 97, 109, 117, 114, 97, 105],
                [9600, 9601, 9602, 9603],
            ]
        )

        tokenizer = UnicodeCodepointTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(
            detokenize_output,
            [
                b"ninja",
                b"samurai",
                b"\xe2\x96\x80\xe2\x96\x81\xe2\x96\x82\xe2\x96\x83",
            ],
        )

    def test_detokenize_replace_error(self):
        # 10000000 is an invalid value
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(
            errors="replace", replacement_char=75
        )
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"niKnja"])

    def test_detokenize_ignore_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"ninja"])

    def test_detokenize_strict_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(errors="strict")
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _ = tokenizer.detokenize(input_data)

    def test_normalization_without_UTF8_valueerror(self):
        with self.assertRaises(ValueError):
            _ = UnicodeCodepointTokenizer(
                errors="strict",
                input_encoding="UTF-16",
                normalization_form="NFC",
            )

    def test_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = UnicodeCodepointTokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[110, 105, 110, 106, 97, 115]],
        )

    def test_skip_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = UnicodeCodepointTokenizer(lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[78, 105, 78, 74, 97, 83]],
        )

    def test_tokenize_first_batch_second(self):
        tokenizer = UnicodeCodepointTokenizer()

        ds = tf.data.Dataset.from_tensor_slices(
            ["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"]
        )
        ds = ds.map(tokenizer)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(5))
        output = ds.take(1).get_single_element()

        exp_output = [
            [110, 105, 110, 106, 97],
            [115, 97, 109, 117, 114, 97, 105],
            [9600, 9601, 9602, 9603],
            [107, 101, 114, 97, 115],
            [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
        ]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_tokenize_first_batch_second_with_sequence_length(self):
        tokenizer = UnicodeCodepointTokenizer(sequence_length=10)

        ds = tf.data.Dataset.from_tensor_slices(
            ["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"]
        )
        ds = ds.map(tokenizer)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(5))
        output = ds.take(1).get_single_element()

        exp_output = [
            [110, 105, 110, 106, 97, 0, 0, 0, 0, 0],
            [115, 97, 109, 117, 114, 97, 105, 0, 0, 0],
            [9600, 9601, 9602, 9603, 0, 0, 0, 0, 0, 0],
            [107, 101, 114, 97, 115, 0, 0, 0, 0, 0],
            [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
        ]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_batch_first_tokenize_second(self):
        tokenizer = UnicodeCodepointTokenizer()

        ds = tf.data.Dataset.from_tensor_slices(
            ["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"]
        )
        ds = ds.batch(5).map(tokenizer)
        output = ds.take(1).get_single_element()

        exp_output = [
            [110, 105, 110, 106, 97],
            [115, 97, 109, 117, 114, 97, 105],
            [9600, 9601, 9602, 9603],
            [107, 101, 114, 97, 115],
            [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
        ]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_batch_first_tokenize_second_with_sequence_length(self):
        tokenizer = UnicodeCodepointTokenizer(sequence_length=10)

        ds = tf.data.Dataset.from_tensor_slices(
            ["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"]
        )
        ds = ds.batch(5).map(tokenizer)
        output = ds.take(1).get_single_element()

        exp_output = [
            [110, 105, 110, 106, 97, 0, 0, 0, 0, 0],
            [115, 97, 109, 117, 114, 97, 105, 0, 0, 0],
            [9600, 9601, 9602, 9603, 0, 0, 0, 0, 0, 0],
            [107, 101, 114, 97, 115, 0, 0, 0, 0, 0],
            [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
        ]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_functional_model(self):
        input_data = tf.constant(
            ["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"]
        )
        tokenizer = UnicodeCodepointTokenizer()
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(
            model_output,
            [
                b"ninja",
                b"samurai",
                b"\xe2\x96\x80\xe2\x96\x81\xe2\x96\x82\xe2\x96\x83",
                b"keras",
                b"tensorflow",
            ],
        )

    def test_load_model_with_config(self):
        input_data = tf.constant(["hello"])

        original_tokenizer = UnicodeCodepointTokenizer(
            lowercase=False,
            sequence_length=11,
            normalization_form="NFC",
            errors="strict",
            vocabulary_size=None,
        )
        cloned_tokenizer = UnicodeCodepointTokenizer.from_config(
            original_tokenizer.get_config()
        )
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

        decoded_input = [107, 101, 114, 97, 115]
        self.assertAllEqual(
            original_tokenizer.detokenize(decoded_input),
            cloned_tokenizer.detokenize(decoded_input),
        )

    def test_config(self):
        tokenizer = UnicodeCodepointTokenizer(
            name="unicode_character_tokenizer_config_gen",
            lowercase=False,
            sequence_length=8,
            normalization_form="NFC",
            errors="ignore",
            replacement_char=0,
            vocabulary_size=100,
        )
        exp_config = {
            "dtype": "int32",
            "errors": "ignore",
            "lowercase": False,
            "name": "unicode_character_tokenizer_config_gen",
            "normalization_form": "NFC",
            "replacement_char": 0,
            "sequence_length": 8,
            "input_encoding": "UTF-8",
            "output_encoding": "UTF-8",
            "trainable": True,
            "vocabulary_size": 100,
        }
        self.assertEqual(tokenizer.get_config(), exp_config)

        tokenize_different_encoding = UnicodeCodepointTokenizer(
            name="unicode_character_tokenizer_config_gen",
            lowercase=False,
            sequence_length=8,
            errors="ignore",
            replacement_char=0,
            input_encoding="UTF-16",
            output_encoding="UTF-16",
            vocabulary_size=None,
        )
        exp_config_different_encoding = {
            "dtype": "int32",
            "errors": "ignore",
            "lowercase": False,
            "name": "unicode_character_tokenizer_config_gen",
            "normalization_form": None,
            "replacement_char": 0,
            "sequence_length": 8,
            "input_encoding": "UTF-16",
            "output_encoding": "UTF-16",
            "trainable": True,
            "vocabulary_size": None,
        }
        self.assertEqual(
            tokenize_different_encoding.get_config(),
            exp_config_different_encoding,
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["ninjas and samurais", "time travel"])

        tokenizer = UnicodeCodepointTokenizer(
            name="unicode_character_tokenizer_config_gen",
            lowercase=False,
            sequence_length=20,
            normalization_form="NFKC",
            errors="replace",
            vocabulary_size=None,
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
