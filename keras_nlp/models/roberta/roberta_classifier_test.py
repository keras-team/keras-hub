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
"""Tests for RoBERTa classification model."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.roberta.roberta_backbone import RobertaBackbone
from keras_nlp.models.roberta.roberta_classifier import RobertaClassifier
from keras_nlp.models.roberta.roberta_preprocessor import RobertaPreprocessor
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer


class RobertaClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.backbone = RobertaBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
        )
        self.vocab = {
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
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]
        self.merges = merges
        self.preprocessor = RobertaPreprocessor(
            RobertaTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.classifier = RobertaClassifier(
            self.backbone,
            4,
            preprocessor=self.preprocessor,
        )
        self.classifier_no_preprocessing = RobertaClassifier(
            self.backbone,
            4,
            preprocessor=None,
        )

        self.raw_batch = tf.constant(
            [
                " airplane at airport",
                " the airplane is the best",
                " the best airport",
                " kohli is the best",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (self.raw_batch, tf.ones((4,)))
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_predict(self, jit_compile):
        self.classifier.compile(jit_compile=jit_compile)
        self.classifier.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_predict_no_preprocessing(self, jit_compile):
        self.classifier_no_preprocessing.compile(jit_compile=jit_compile)
        self.classifier_no_preprocessing.predict(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_fit(self, jit_compile):
        self.classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.classifier.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_fit_no_preprocessing(self, jit_compile):
        self.classifier_no_preprocessing.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.classifier_no_preprocessing.fit(self.preprocessed_dataset)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saving_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.classifier.save(save_path, save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, RobertaClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
