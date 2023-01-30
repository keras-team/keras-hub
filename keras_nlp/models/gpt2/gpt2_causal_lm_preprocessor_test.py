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
            "<|endoftext|>": 0,
            "!": 1,
            "air": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
            "Ġkoh": 7,
            "li": 8,
            "Ġis": 9,
            "Ġthe": 10,
            "Ġbest": 11,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "ai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        self.merges = merges

        self.preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
        )

    def test_tokenize_strings(self):
        input_data = "airplane at airport"

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [2, 4, 5, 3, 6, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0])
        self.assertAllEqual(y, [4, 5, 3, 6, 0, 0, 0])
        self.assertAllEqual(sw, [1, 1, 1, 1, 0, 0, 0])

    def test_tokenize_list_of_strings(self):
        input_data = ["airplane at airport"] * 4

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [[2, 4, 5, 3, 6, 0, 0]] * 4,
        )
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[4, 5, 3, 6, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_pad_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [[0, 2, 4, 5, 3, 6, 0]] * 4,
        )
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1]] * 4)
        self.assertAllEqual(y, [[2, 4, 5, 3, 6, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0]] * 4)

    def test_tokenize_labeled_batch(self):
        x = tf.constant(["airplane at airport"] * 4)
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        x, y, sw = self.preprocessor(x, y, sw)
        self.assertAllEqual(
            x["token_ids"],
            [[2, 4, 5, 3, 6, 0, 0]] * 4,
        )
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[4, 5, 3, 6, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_tokenize_labeled_dataset(self):
        x = tf.constant(["airplane at airport"] * 4)
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        ds = tf.data.Dataset.from_tensor_slices((x, y, sw))
        ds = ds.map(self.preprocessor)
        x, y, sw = ds.batch(4).take(1).get_single_element()

        self.assertAllEqual(
            x["token_ids"],
            [[2, 4, 5, 3, 6, 0, 0]] * 4,
        )
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[4, 5, 3, 6, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0]] * 4)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs, y, sw = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
