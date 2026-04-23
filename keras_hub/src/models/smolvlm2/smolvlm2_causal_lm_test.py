from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm import SmolVLM2CausalLM
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    SmolVLM2CausalLMPreprocessor,
)
from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class SmolVLM2CausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|begin_of_text|>"]
        self.vocab += ["<|end_of_text|>"]
        self.vocab += ["<image>"]
        self.vocab += ["<end_of_utterance>"]
        self.vocab += ["<|im_start|>"]
        self.vocab += ["<|im_end|>"]
        self.vocab += ["<fake_token_around_image>"]
        self.vocab += ["<global-img>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.preprocessor = SmolVLM2CausalLMPreprocessor(
            SmolVLM2Tokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.backbone = SmolVLM2Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            image_size=32,
            patch_size=16,
            vision_hidden_dim=64,
            vision_intermediate_dim=128,
            vision_num_layers=2,
            vision_num_heads=4,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            scale_factor=1,
            image_token_id=8,  # <image> token id
            rope_max_wavelength=10000,
            layer_norm_epsilon=1e-5,
            vision_layer_norm_epsilon=1e-6,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=SmolVLM2CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                8,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate(self):
        causal_lm = SmolVLM2CausalLM(**self.init_kwargs)
        # String input.
        prompt = " airplane at airport"
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
        causal_lm = SmolVLM2CausalLM(**self.init_kwargs)
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
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = SmolVLM2CausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SmolVLM2CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_score_logits(self):
        prompts = [" airplane at airport", " airplane at airport"]
        causal_lm = SmolVLM2CausalLM(**self.init_kwargs)
        expected_score_shape = (
            2,
            8,
            self.preprocessor.tokenizer.vocabulary_size(),
        )

        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_loss(self):
        prompts = [" airplane at airport", " airplane at airport"]
        causal_lm = SmolVLM2CausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8)

        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]
        target_ids = ops.roll(token_ids, shift=-1, axis=1)

        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="loss",
            target_ids=target_ids,
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)
