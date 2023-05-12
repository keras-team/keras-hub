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

"""Tests for GPT2 causal LM preprocessor layer."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer


class GPT2CausalLMPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.vocab = {
            "!": 0,
            "air": 1,
            "Ġair": 2,
            "plane": 3,
            "Ġat": 4,
            "port": 5,
            "<|endoftext|>": 6,
        }

        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]

        self.preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
        )

    def test_strings(self):
        input_data = "airplane at airport"

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 4, 2, 5, 6, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0])
        self.assertAllEqual(y, [1, 3, 4, 2, 5, 6, 0, 0])
        self.assertAllEqual(sw, [1, 1, 1, 1, 1, 1, 0, 0])

    def test_list_of_strings(self):
        input_data = ["airplane at airport"] * 4

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[1, 3, 4, 2, 5, 6, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_no_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 2, 5, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 2, 5, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_labeled_batch(self):
        x = tf.constant(["airplane at airport"] * 4)
        y = tf.constant([1] * 4)  # Ignored.
        sw = tf.constant([1.0] * 4)  # Ignored.
        x, y, sw = self.preprocessor(x, y, sw)
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[1, 3, 4, 2, 5, 6, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_dataset(self):
        x = tf.constant(["airplane at airport"] * 4)
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(self.preprocessor)
        x, y, sw = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[1, 3, 4, 2, 5, 6, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        x = self.preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 4, 2, 5, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": tf.constant([6, 1, 3, 4, 2, 5, 0, 0]),
            "padding_mask": tf.cast([1, 1, 1, 1, 1, 1, 0, 0], dtype="bool"),
        }
        x = self.preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.preprocessor)
        new_preprocessor = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_preprocessor.get_config(),
            self.preprocessor.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs, y, sw = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
