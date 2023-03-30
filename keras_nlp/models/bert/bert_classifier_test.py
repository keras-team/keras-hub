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
"""Tests for BERT classification model."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_classifier import BertClassifier
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer


class BertClassifierTest(tf.test.TestCase, parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup model.
        cls.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        cls.vocab += ["the", "quick", "brown", "fox", "."]
        cls.preprocessor = BertPreprocessor(
            BertTokenizer(vocabulary=cls.vocab),
            sequence_length=5,
        )
        cls.backbone = BertBackbone(
            vocabulary_size=cls.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=cls.preprocessor.packer.sequence_length,
        )
        cls.classifier = BertClassifier(
            cls.backbone,
            4,
            preprocessor=cls.preprocessor,
        )

        # Setup data.
        cls.raw_batch = tf.constant(
            [
                "the quick brown fox.",
                "the slow brown fox.",
            ]
        )
        cls.preprocessed_batch = cls.preprocessor(cls.raw_batch)
        cls.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (cls.raw_batch, tf.ones((2,)))
        ).batch(2)
        cls.preprocessed_dataset = cls.raw_dataset.map(cls.preprocessor)

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    def test_classifier_predict(self):
        self.classifier.predict(self.raw_batch)
        self.classifier.preprocessor = None
        self.classifier.predict(self.preprocessed_batch)
        self.classifier.preprocessor = self.preprocessor

    def test_classifier_fit(self):
        self.classifier.fit(self.raw_dataset)
        self.classifier.preprocessor = None
        self.classifier.fit(self.preprocessed_dataset)
        self.classifier.preprocessor = self.preprocessor

    def test_classifier_fit_no_xla(self):
        self.classifier.preprocessor = None
        self.classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.classifier.fit(self.preprocessed_dataset)
        self.classifier.preprocessor = self.preprocessor

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.classifier)
        new_classifier = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_classifier.get_config(),
            self.classifier.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.classifier.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BertClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
