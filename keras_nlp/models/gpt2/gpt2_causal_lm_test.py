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
from unittest.mock import patch

import numpy as np
import pytest
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
        # For DTensor.
        keras.backend.experimental.enable_tf_random_generator()
        keras.utils.set_random_seed(1337)

        self.vocab = {
            "!": 0,
            "air": 1,
            "Ġair": 2,
            "plane": 3,
            "Ġat": 4,
            "port": 5,
            "<|endoftext|>": 6,
        }
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.preprocessor = GPT2CausalLMPreprocessor(
            GPT2Tokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.backbone = GPT2Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )
        self.causal_lm = GPT2CausalLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

        self.raw_batch = tf.constant(
            [
                " airplane at airport",
                " airplane at airport",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)[0]
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            self.raw_batch
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_causal_lm(self):
        self.causal_lm(self.preprocessed_batch)

    def test_predict(self):
        self.causal_lm.predict(self.raw_batch)
        self.causal_lm.preprocessor = None
        self.causal_lm.predict(self.preprocessed_batch)

    def test_fit(self):
        self.causal_lm.fit(self.raw_dataset)
        self.causal_lm.preprocessor = None
        self.causal_lm.fit(self.preprocessed_dataset)

    def test_fit_no_xla(self):
        self.causal_lm.preprocessor = None
        self.causal_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.causal_lm.fit(self.preprocessed_dataset)

    def test_generate(self):
        # String input.
        prompt = " airplane at airport"
        output = self.causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        # String tensor input.
        self.assertIsInstance(self.causal_lm.generate(self.raw_batch)[0], str)
        # String dataset input.
        self.assertIsInstance(self.causal_lm.generate(self.raw_dataset)[0], str)
        # Int tensor input.
        self.causal_lm.preprocessor = None
        outputs = self.causal_lm.generate(self.preprocessed_batch)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            self.preprocessed_batch["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            self.preprocessed_batch["padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        call_with_cache = self.causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            logits = np.zeros(logits.shape.as_list())
            logits[:, :, self.preprocessor.tokenizer.end_token_id] = 1.0e9
            return logits, hidden_states, cache

        with patch.object(self.causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = self.causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)
            self.assertEqual(self.causal_lm.call_with_cache.call_count, 2)

    def test_generate_compilation(self):
        # Assert we do not recompile with successive calls.
        self.causal_lm.generate(self.raw_batch)
        first_fn = self.causal_lm.generate_function
        self.causal_lm.generate(self.raw_batch)
        second_fn = self.causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        self.causal_lm.compile(sampler="greedy")
        self.assertIsNone(self.causal_lm.generate_function)

    def test_serialization(self):
        new_causal_lm = keras.utils.deserialize_keras_object(
            keras.utils.serialize_keras_object(self.causal_lm)
        )
        self.assertEqual(
            new_causal_lm.get_config(), self.causal_lm.get_config()
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        keras.utils.set_random_seed(42)
        model_output = self.causal_lm.predict(self.raw_batch)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        self.causal_lm.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, GPT2CausalLM)

        # Check that output matches.
        keras.utils.set_random_seed(42)
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_layout_map(self):
        mesh = tf.experimental.dtensor.create_mesh([("batch", 1), ("model", 1)])
        with GPT2CausalLM.create_layout_map(mesh).scope():
            GPT2CausalLM(backbone=self.backbone)
        # Using DTensor enables the mlir bridge as a side effect. Eventually
        # this will be default, but for now we have compile errors with the
        # bridge elsewhere and must disable. See
        # https://github.com/keras-team/keras-nlp/issues/1001
        tf.config.experimental.disable_mlir_bridge()
