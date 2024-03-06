# Copyright 2024 The KerasNLP Authors
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

from unittest.mock import patch

import pytest

from keras_nlp.backend import ops
from keras_nlp.models.bloom.bloom_backbone import BloomBackbone
from keras_nlp.models.bloom.bloom_causal_lm import BloomCausalLM
from keras_nlp.models.bloom.bloom_causal_lm_preprocessor import (
    BloomCausalLMPreprocessor,
)
from keras_nlp.models.bloom.bloom_tokenizer import BloomTokenizer
from keras_nlp.tests.test_case import TestCase


class BloomCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["<unk>", "<s>", "</s>", "<pad>"]
        self.vocab += ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = BloomTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.preprocessor = BloomCausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        self.backbone = BloomBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            [
                " airplane at airport",
                " airplane airport",
            ],
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        vocabulary_size = self.tokenizer.vocabulary_size()
        self.run_task_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, vocabulary_size),
        )

    def test_generate(self):
        causal_lm = BloomCausalLM(**self.init_kwargs)
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_generate_with_bfloat16(self):
        backbone = BloomBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "bfloat16"}
        )
        causal_lm = BloomCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_generate_with_mixed_float16(self):
        backbone = BloomBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "mixed_float16"}
        )
        causal_lm = BloomCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_early_stopping(self):
        causal_lm = BloomCausalLM(**self.init_kwargs)
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
            prompt = ["airplane at", "airplane"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = BloomCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate("airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BloomCausalLM.presets:
            self.run_preset_test(
                cls=BloomCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
