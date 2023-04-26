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
"""Tests for ALBERT classification model."""

import io
import os

import pytest
import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_classifier import AlbertClassifier
from keras_nlp.models.albert.albert_preprocessor import AlbertPreprocessor
from keras_nlp.models.albert.albert_tokenizer import AlbertTokenizer


class AlbertClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        # Setup model

        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=10,
            model_type="WORD",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            user_defined_symbols="[MASK]",
        )
        self.proto = bytes_io.getvalue()

        tokenizer = AlbertTokenizer(proto=self.proto)

        self.preprocessor = AlbertPreprocessor(
            tokenizer=tokenizer,
            sequence_length=5,
        )
        self.backbone = AlbertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            embedding_dim=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )

        self.classifier = AlbertClassifier(
            self.backbone,
            num_classes=4,
            preprocessor=self.preprocessor,
            # Check we handle serialization correctly.
            activation=keras.activations.softmax,
        )

        self.raw_batch = tf.constant(
            [
                "the quick brown fox.",
                "the slow brown fox.",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (self.raw_batch, tf.ones((2,)))
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    def test_classifier_predict(self):
        preds1 = self.classifier.predict(self.raw_batch)
        self.classifier.preprocessor = None
        preds2 = self.classifier.predict(self.preprocessed_batch)
        # Assert predictions match.
        self.assertAllClose(preds1, preds2)
        # Assert valid softmax output.
        self.assertAllClose(tf.reduce_sum(preds2, axis=-1), [1.0, 1.0])

    def test_classifier_fit(self):
        self.classifier.fit(self.raw_dataset)
        self.classifier.preprocessor = None
        self.classifier.fit(self.preprocessed_dataset)

    def test_classifier_fit_no_xla(self):
        self.classifier.preprocessor = None
        self.classifier.compile(
            loss="sparse_categorical_crossentropy",
            jit_compile=False,
        )
        self.classifier.fit(self.preprocessed_dataset)

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
    @pytest.mark.large
    def test_saving_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        self.classifier.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)

        # Check we got the real object back
        self.assertIsInstance(restored_model, AlbertClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
