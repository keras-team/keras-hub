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
"""Tests for RoBERTa masked language model."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.roberta.roberta_backbone import RobertaBackbone
from keras_nlp.models.roberta.roberta_masked_lm import RobertaMaskedLM
from keras_nlp.models.roberta.roberta_masked_lm_preprocessor import (
    RobertaMaskedLMPreprocessor,
)
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer


class RobertaMaskedLMTest(tf.test.TestCase, parameterized.TestCase):
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
            "<mask>": 12,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]
        self.merges = merges
        self.preprocessor = RobertaMaskedLMPreprocessor(
            RobertaTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
            mask_selection_length=2,
        )
        self.masked_lm = RobertaMaskedLM(
            self.backbone,
            preprocessor=self.preprocessor,
        )
        self.masked_lm_no_preprocessing = RobertaMaskedLM(
            self.backbone,
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
    def test_roberta_masked_lm_predict(self, jit_compile):
        self.masked_lm.compile(jit_compile=jit_compile)
        self.masked_lm.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_masked_lm_predict_no_preprocessing(self, jit_compile):
        self.masked_lm_no_preprocessing.compile(jit_compile=jit_compile)
        self.masked_lm_no_preprocessing.predict(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_masked_lm_fit(self, jit_compile):
        self.masked_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.masked_lm.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_masked_lm_fit_no_preprocessing(self, jit_compile):
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
        self.assertIsInstance(restored_model, RobertaMaskedLM)

        model_output = self.masked_lm(self.preprocessed_batch)
        restored_output = restored_model(self.preprocessed_batch)

        self.assertAllClose(model_output, restored_output)
