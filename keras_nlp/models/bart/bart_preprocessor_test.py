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

"""Tests for BART preprocessor layer."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bart.bart_preprocessor import BartPreprocessor
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer


class BartPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        vocab = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
            "Ġkoh": 7,
            "li": 8,
            "Ġis": 9,
            "Ġthe": 10,
            "Ġbest": 11,
            "<mask>": 12,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]

        self.preprocessor = BartPreprocessor(
            tokenizer=BartTokenizer(
                vocabulary=vocab,
                merges=merges,
            ),
            encoder_sequence_length=10,
            decoder_sequence_length=9,
        )

    def test_tokenize_strings(self):
        input_data = {
            "encoder_text": " airplane at airport",
            "decoder_text": " kohli is the best",
        }

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["encoder_token_ids"], [0, 3, 4, 5, 3, 6, 2, 1, 1, 1]
        )
        self.assertAllEqual(
            output["encoder_padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        )
        self.assertAllEqual(
            output["decoder_token_ids"], [2, 0, 7, 8, 9, 10, 11, 2, 1]
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 0]
        )

    def test_tokenize_list_of_strings(self):
        input_data = {
            "encoder_text": [" airplane at airport"] * 4,
            "decoder_text": [" kohli is the best"] * 4,
        }

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["encoder_token_ids"], [[0, 3, 4, 5, 3, 6, 2, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            output["encoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            output["decoder_token_ids"], [[2, 0, 7, 8, 9, 10, 11, 2, 1]] * 4
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0]] * 4
        )

    def test_tokenize_labeled_batch(self):
        x = {
            "encoder_text": [" airplane at airport"] * 4,
            "decoder_text": [" kohli is the best"] * 4,
        }
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        x_out, y_out, sw_out = self.preprocessor(x, y, sw)
        self.assertAllEqual(
            x_out["encoder_token_ids"], [[0, 3, 4, 5, 3, 6, 2, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            x_out["encoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            x_out["decoder_token_ids"], [[2, 0, 7, 8, 9, 10, 11, 2, 1]] * 4
        )
        self.assertAllEqual(
            x_out["decoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0]] * 4
        )
        self.assertAllEqual(y_out, y)
        self.assertAllEqual(sw_out, sw)

    def test_tokenize_labeled_dataset(self):
        x = {
            "encoder_text": [" airplane at airport"] * 4,
            "decoder_text": [" kohli is the best"] * 4,
        }
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        ds = tf.data.Dataset.from_tensor_slices((x, y, sw))
        ds = ds.map(self.preprocessor)
        x_out, y_out, sw_out = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(
            x_out["encoder_token_ids"], [[0, 3, 4, 5, 3, 6, 2, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            x_out["encoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            x_out["decoder_token_ids"], [[2, 0, 7, 8, 9, 10, 11, 2, 1]] * 4
        )
        self.assertAllEqual(
            x_out["decoder_padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0]] * 4
        )
        self.assertAllEqual(y_out, y)
        self.assertAllEqual(sw_out, sw)

    def test_error_multi_segment_input(self):
        input_data = {
            "encoder_text": (
                tf.constant([" airplane at airport"] * 2),
                tf.constant([" airplane"] * 2),
            ),
            "decoder_text": (
                tf.constant([" kohli is the best"] * 2),
                tf.constant([" kohli"] * 2),
            ),
        }

        with self.assertRaises(ValueError):
            self.preprocessor(input_data)

    def test_serialization(self):
        new_preprocessor = keras.utils.deserialize_keras_object(
            keras.utils.serialize_keras_object(self.preprocessor)
        )
        self.assertEqual(
            new_preprocessor.get_config(), self.preprocessor.get_config()
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        input_data = {
            "encoder_text": tf.constant(" airplane at airport"),
            "decoder_text": tf.constant(" kohli is the best"),
        }

        inputs = {
            "encoder_text": keras.Input(dtype="string", shape=()),
            "decoder_text": keras.Input(dtype="string", shape=()),
        }
        outputs = self.preprocessor(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)

        restored_model = keras.models.load_model(path)

        model_output = model(input_data)
        restored_model_output = restored_model(input_data)

        self.assertAllClose(
            model_output,
            restored_model_output,
        )
