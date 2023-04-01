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

"""Tests for RoBERTa masked language model preprocessor layer."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.roberta.roberta_masked_lm_preprocessor import (
    RobertaMaskedLMPreprocessor,
)
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer


class RobertaMaskedLMPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
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

        self.preprocessor = RobertaMaskedLMPreprocessor(
            tokenizer=RobertaTokenizer(
                vocabulary=vocab,
                merges=merges,
            ),
            # Simplify out testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=12,
        )

    def test_preprocess_strings(self):
        input_data = " airplane at airport"

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [0, 12, 12, 12, 12, 12, 2, 1, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )
        self.assertAllEqual(x["mask_positions"], [1, 2, 3, 4, 5])
        self.assertAllEqual(y, [3, 4, 5, 3, 6])
        self.assertAllEqual(sw, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_preprocess_list_of_strings(self):
        input_data = [" airplane at airport"] * 4

        x, y, sw = self.preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [[0, 12, 12, 12, 12, 12, 2, 1, 1, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(x["mask_positions"], [[1, 2, 3, 4, 5]] * 4)
        self.assertAllEqual(y, [[3, 4, 5, 3, 6]] * 4)
        self.assertAllEqual(sw, [[1.0, 1.0, 1.0, 1.0, 1.0]] * 4)

    def test_preprocess_dataset(self):
        sentences = tf.constant([" airplane at airport"] * 4)
        ds = tf.data.Dataset.from_tensor_slices(sentences)
        ds = ds.map(self.preprocessor)
        x, y, sw = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(
            x["token_ids"], [[0, 12, 12, 12, 12, 12, 2, 1, 1, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(x["mask_positions"], [[1, 2, 3, 4, 5]] * 4)
        self.assertAllEqual(y, [[3, 4, 5, 3, 6]] * 4)
        self.assertAllEqual(sw, [[1.0, 1.0, 1.0, 1.0, 1.0]] * 4)

    def test_mask_multiple_sentences(self):
        sentence_one = tf.constant(" airplane")
        sentence_two = tf.constant(" kohli")

        x, y, sw = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            x["token_ids"], [0, 12, 12, 2, 2, 12, 12, 2, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        )
        self.assertAllEqual(x["mask_positions"], [1, 2, 5, 6, 0])
        self.assertAllEqual(y, [3, 4, 7, 8, 0])
        self.assertAllEqual(sw, [1.0, 1.0, 1.0, 1.0, 0.0])

    def test_no_masking_zero_rate(self):
        no_mask_preprocessor = RobertaMaskedLMPreprocessor(
            self.preprocessor.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=5,
            sequence_length=12,
        )
        input_data = " airplane at airport"

        x, y, sw = no_mask_preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [0, 3, 4, 5, 3, 6, 2, 1, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )
        self.assertAllEqual(x["mask_positions"], [0, 0, 0, 0, 0])
        self.assertAllEqual(y, [0, 0, 0, 0, 0])
        self.assertAllEqual(sw, [0.0, 0.0, 0.0, 0.0, 0.0])

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
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)

        restored_model = keras.models.load_model(path)
        outputs = model(input_data)[0]["token_ids"]
        restored_outputs = restored_model(input_data)[0]["token_ids"]
        self.assertAllEqual(outputs, restored_outputs)
