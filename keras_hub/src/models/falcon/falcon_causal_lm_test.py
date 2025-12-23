from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.models.falcon.falcon_causal_lm import FalconCausalLM
from keras_hub.src.models.falcon.falcon_causal_lm_preprocessor import (
    FalconCausalLMPreprocessor,
)
from keras_hub.src.models.falcon.falcon_tokenizer import FalconTokenizer
from keras_hub.src.tests.test_case import TestCase


class FalconCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = FalconTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.preprocessor = FalconCausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        self.backbone = FalconBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_attention_heads=2,
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
            cls=FalconCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, vocabulary_size),
        )

    def test_generate(self):
        causal_lm = FalconCausalLM(**self.init_kwargs)
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
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
        backbone = FalconBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "bfloat16"}
        )
        causal_lm = FalconCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
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
        backbone = FalconBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "mixed_float16"}
        )
        causal_lm = FalconCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
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
        causal_lm = FalconCausalLM(**self.init_kwargs)
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
        causal_lm = FalconCausalLM(**self.init_kwargs)
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
            cls=FalconCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=FalconCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FalconCausalLM.presets:
            self.run_preset_test(
                cls=FalconCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
