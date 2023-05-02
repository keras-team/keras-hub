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
"""Tests for BERT masked language model."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_masked_lm import BertMaskedLM
from keras_nlp.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer


class BertMaskedLMTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = BertMaskedLMPreprocessor(
            BertTokenizer(vocabulary=self.vocab),
            # Simplify out testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=2,
            sequence_length=5,
        )
        self.backbone = BertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )
        self.masked_lm = BertMaskedLM(
            self.backbone,
            preprocessor=self.preprocessor,
        )

        # Setup data.
        self.raw_batch = tf.constant(
            [
                "the quick brown fox.",
                "the slow brown fox.",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            self.raw_batch
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_classifier(self):
        self.masked_lm(self.preprocessed_batch[0])

    def test_classifier_predict(self):
        self.masked_lm.predict(self.raw_batch)
        self.masked_lm.preprocessor = None
        self.masked_lm.predict(self.preprocessed_batch[0])

    def test_classifier_fit(self):
        self.masked_lm.fit(self.raw_dataset)
        self.masked_lm.preprocessor = None
        self.masked_lm.fit(self.preprocessed_dataset)

    def test_classifier_fit_no_xla(self):
        self.masked_lm.preprocessor = None
        self.masked_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.masked_lm.fit(self.preprocessed_dataset)

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.masked_lm)
        new_classifier = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_classifier.get_config(),
            self.masked_lm.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model_output = self.masked_lm.predict(self.raw_batch)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        self.masked_lm.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BertMaskedLM)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output, atol=0.01, rtol=0.01)
