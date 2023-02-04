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
"""Tests for GPT2 causal LM model."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer


class GPT2CausalLMTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        backbone = GPT2Backbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
        )
        vocab = {
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

        self.preprocessor = GPT2CausalLMPreprocessor(
            GPT2Tokenizer(vocabulary=vocab, merges=merges),
            sequence_length=8,
        )
        self.causal_lm = GPT2CausalLM(
            backbone,
            preprocessor=self.preprocessor,
        )
        self.causal_lm_no_preprocessing = GPT2CausalLM(
            backbone,
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

    def test_valid_call_causal_lm(self):
        self.causal_lm(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_causal_lm_predict(self, jit_compile):
        self.causal_lm.compile(jit_compile=jit_compile)
        self.causal_lm.predict(self.raw_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_causal_lm_predict_no_preprocessing(self, jit_compile):
        self.causal_lm_no_preprocessing.compile(jit_compile=jit_compile)
        self.causal_lm_no_preprocessing.predict(self.preprocessed_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_causal_lm_fit(self, jit_compile):
        self.causal_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.causal_lm.fit(self.raw_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_causal_lm_fit_no_preprocessing(self, jit_compile):
        self.causal_lm_no_preprocessing.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=jit_compile,
        )
        self.causal_lm_no_preprocessing.fit(self.preprocessed_dataset)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_causal_lm_generate(self, jit_compile):
        self.causal_lm_no_preprocessing.compile(jit_compile=jit_compile)
        self.causal_lm.generate(
            self.raw_batch,
            max_length=10,
        )

        # String input
        prompt = " airplane"
        generated = self.causal_lm.generate(
            prompt,
            max_length=10,
        )
        generated = generated.numpy().decode("utf-8")
        self.assertTrue(prompt in generated)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        keras.utils.set_random_seed(42)
        model_output = self.causal_lm.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.causal_lm.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, GPT2CausalLM)

        # Check that output matches.
        keras.utils.set_random_seed(42)
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
