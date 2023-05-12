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

"""Tests for OPT causal LM preprocessor layer."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.opt.opt_causal_lm_preprocessor import (
    OPTCausalLMPreprocessor,
)
from keras_nlp.models.opt.opt_tokenizer import OPTTokenizer


class OPTCausalLMPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.vocab = {
            "<pad>": 0,
            "</s>": 1,
            "air": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]
        self.merges = merges

        self.preprocessor = OPTCausalLMPreprocessor(
            tokenizer=OPTTokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
        )

    def test_strings(self):
        input_data = " airplane at airport"

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 5, 3, 6, 1, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0])
        self.assertAllEqual(y, [3, 4, 5, 3, 6, 1, 0, 0])
        self.assertAllEqual(sw, [1, 1, 1, 1, 1, 1, 0, 0])

    def test_list_of_strings(self):
        input_data = [" airplane at airport"] * 4

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 5, 3, 6, 1, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 5, 3, 6, 1, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_no_start_end_token(self):
        input_data = [" airplane at airport"] * 4

        preprocessor = OPTCausalLMPreprocessor(
            tokenizer=OPTTokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[3, 4, 5, 3, 6, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[4, 5, 3, 6, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_labeled_batch(self):
        x = tf.constant([" airplane at airport"] * 4)
        y = tf.constant([1] * 4)  # Ignored.
        sw = tf.constant([1.0] * 4)  # Ignored.
        x, y, sw = self.preprocessor(x, y, sw)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 5, 3, 6, 1, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 5, 3, 6, 1, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_dataset(self):
        x = tf.constant([" airplane at airport"] * 4)
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(self.preprocessor)
        x, y, sw = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 5, 3, 6, 1, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 5, 3, 6, 1, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = " airplane at airport"
        x = self.preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 5, 3, 6, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": tf.constant([1, 3, 4, 5, 3, 6, 0, 0]),
            "padding_mask": tf.cast([1, 1, 1, 1, 1, 1, 0, 0], dtype="bool"),
        }
        x = self.preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, " airplane at airport")

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
        input_data = tf.constant([" airplane at airport"])

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
