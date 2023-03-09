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
"""Tests for ALBERT masked language model."""

import io
import os

import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_masked_lm import AlbertMaskedLM
from keras_nlp.models.albert.albert_masked_lm_preprocessor import (
    AlbertMaskedLMPreprocessor,
)
from keras_nlp.models.albert.albert_tokenizer import AlbertTokenizer


class AlbertMaskedLMTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.backbone = AlbertBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            embedding_dim=128,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
        )
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round", "an eagle flew"]
        )

        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=15,
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

        proto = bytes_io.getvalue()

        tokenizer = AlbertTokenizer(proto=proto)

        self.preprocessor = AlbertMaskedLMPreprocessor(
            tokenizer=tokenizer,
            # Simplify out testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=12,
        )
        self.masked_lm = AlbertMaskedLM(
            self.backbone,
            preprocessor=self.preprocessor,
        )
        self.masked_lm_no_preprocessing = AlbertMaskedLM(
            self.backbone,
            preprocessor=None,
        )

        self.raw_batch = tf.constant(
            [
                "quick brown fox",
                "eagle flew over fox",
                "the eagle flew quick",
                "a brown eagle",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)[0]
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            self.raw_batch
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_masked_lm(self):
        self.masked_lm(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_masked_lm_predict(self, jit_compile):
        self.masked_lm.compile(jit_compile=jit_compile)
        self.masked_lm.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_masked_lm_predict_no_preprocessing(self, jit_compile):
        self.masked_lm_no_preprocessing.compile(jit_compile=jit_compile)
        self.masked_lm_no_preprocessing.predict(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_masked_lm_fit(self, jit_compile):
        self.masked_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.masked_lm.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_albert_masked_lm_fit_no_preprocessing(self, jit_compile):
        self.masked_lm_no_preprocessing.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.masked_lm_no_preprocessing.fit(self.preprocessed_dataset)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.masked_lm.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, AlbertMaskedLM)

        model_output = self.masked_lm(self.preprocessed_batch)
        restored_output = restored_model(self.preprocessed_batch)

        self.assertAllClose(model_output, restored_output)
