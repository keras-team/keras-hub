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
"""Tests for BERT classification model."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_feature_extractor import BertFeatureExtractor
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer


class BertFeatureExtractorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.backbone = BertBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = BertPreprocessor(
            BertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        self.featurizer = BertFeatureExtractor(
            self.backbone, preprocessor=self.preprocessor
        )

        self.featurizer_no_preprocessing = BertFeatureExtractor(
            self.backbone,
            preprocessor=None,
        )

        self.raw_batch = tf.constant(
            [
                "the quick brown fox.",
                "the slow brown fox.",
                "the smelly brown fox.",
                "the old brown fox.",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (self.raw_batch, tf.ones((4,)))
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_featurizer(self):
        self.featurizer(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_predict(self, jit_compile):
        features = self.featurizer(self.preprocessed_batch)["pooled_outputs"]
        classifier = keras.layers.Dense(4)(features)

        classifier.compile(jit_compile=jit_compile)
        classifier.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_predict_no_preprocessing(self, jit_compile):
        features = self.featurizer_no_preprocessing(self.preprocessed_batch)["pooled_outputs"]
        classifier = keras.layers.Dense(4)(features)

        classifier.compile(jit_compile=jit_compile)
        classifier.predict(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_fit(self, jit_compile):
        features = self.featurizer(self.preprocessed_batch)["pooled_outputs"]

        classifier = keras.layers.Dense(4)(features)

        classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        classifier.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_fit_no_preprocessing(self, jit_compile):
        features = self.featurizer_no_preprocessing(self.preprocessed_batch)["pooled_outputs"]

        classifier = keras.layers.Dense(4)(features)

        classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        classifier.fit(self.preprocessed_dataset)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_saved_model(self, save_format, filename, jit_compile):

        features = self.featurizer(self.preprocessed_batch)["pooled_outputs"]

        classifier = keras.layers.Dense(4)(features)        
        classifier.compile(jit_compile=jit_compile)

        model_output = classifier.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.featurizer.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BertFeatureExtractor)

        # Check that output matches.
        restored_output = classifier.predict(restored_model(self.raw_batch))
        self.assertAllClose(model_output, restored_output)
