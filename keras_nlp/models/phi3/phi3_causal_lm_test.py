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

import os
from unittest.mock import patch

from keras_nlp.backend import ops
from keras_nlp.models.phi3.phi3_backbone import Phi3Backbone
from keras_nlp.models.phi3.phi3_causal_lm import Phi3CausalLM
from keras_nlp.models.phi3.phi3_causal_lm_preprocessor import (
    Phi3CausalLMPreprocessor,
)
from keras_nlp.models.phi3.phi3_tokenizer import Phi3Tokenizer
from keras_nlp.tests.test_case import TestCase

# import pytest


class Phi3CausalLMTest(TestCase):
    def setUp(self):
        self.preprocessor = Phi3CausalLMPreprocessor(
            Phi3Tokenizer(
                # Generated using create_phi3_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "phi3_test_vocab.spm"
                )
            ),
            sequence_length=12,
        )
        self.vocab_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = Phi3Backbone(
            vocabulary_size=self.vocab_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 12, self.vocab_size),
        )

    def test_generate(self):
        causal_lm = Phi3CausalLM(**self.init_kwargs)
        # String input.
        prompt = "the fox"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        causal_lm = Phi3CausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the fox", "the earth"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Phi3CausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    # @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    # @pytest.mark.extra_large
    # def test_all_presets(self):
    #     for preset in Phi3CausalLM.presets:
    #         self.run_preset_test(
    #             cls=Phi3CausalLM,
    #             preset=preset,
    #             input_data=self.input_data,
    #         )
