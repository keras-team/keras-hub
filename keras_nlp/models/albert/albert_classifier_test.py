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
        self.backbone = AlbertBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            embedding_dim=8,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )

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
            sequence_length=8,
        )
        self.classifier = AlbertClassifier(
            self.backbone,
            4,
            preprocessor=self.preprocessor,
        )
        self.classifier_no_preprocessing = AlbertClassifier(
            self.backbone,
            4,
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

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_classifier_predict(self, jit_compile):
        self.classifier.compile(jit_compile=jit_compile)
        self.classifier.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_classifier_predict_no_preprocessing(self, jit_compile):
        self.classifier_no_preprocessing.compile(jit_compile=jit_compile)
        self.classifier_no_preprocessing.predict(self.preprocessed_batch)

    def test_albert_classifier_fit_default_compile(self):
        self.classifier.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_classifier_fit(self, jit_compile):
        self.classifier.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.classifier.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_classifier_fit_no_preprocessing(self, jit_compile):
        self.classifier_no_preprocessing.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.classifier_no_preprocessing.fit(self.preprocessed_dataset)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.classifier.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, AlbertClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
