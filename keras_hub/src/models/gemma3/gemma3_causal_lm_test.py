import os
from unittest.mock import patch

import keras
import pytest
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import running_on_gpu


class Gemma3CausalLMTest(TestCase):
    def setUp(self):
        self.tokenizer = Gemma3Tokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            ),
        )
        self.text_preprocessor = Gemma3CausalLMPreprocessor(
            image_converter=None,
            tokenizer=self.tokenizer,
            sequence_length=20,
        )

        text_backbone_init_kwargs = {
            # vocabulary
            "vocabulary_size": (
                self.text_preprocessor.tokenizer.vocabulary_size()
            ),
            # image
            "image_size": 16,
            # model
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            # other model args
            "query_head_dim_normalize": True,
            "use_query_key_norm": True,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": None,
            "attention_logit_soft_cap": None,
            "use_sliding_window_attention": True,
            "sliding_window_size": 1024,
            "vision_encoder": None,
        }

        self.text_backbone = Gemma3Backbone(**text_backbone_init_kwargs)
        self.text_init_kwargs = {
            "preprocessor": self.text_preprocessor,
            "backbone": self.text_backbone,
        }
        self.text_train_data = (
            {
                "prompts": ["the quick brown fox", "the quick brown fox"],
                "responses": ["the earth is round", "the earth is round"],
            },
        )
        self.text_input_data = self.text_preprocessor(*self.text_train_data)[0]

    def test_text_causal_lm_basics(self):
        self.run_task_test(
            cls=Gemma3CausalLM,
            init_kwargs=self.text_init_kwargs,
            train_data=self.text_train_data,
            expected_output_shape=(2, 20, 16),
        )

    def test_text_flash_attention_call(self):
        if (
            keras.config.backend() != "jax"
            or not fused_attention_op_available()
        ):
            self.skipTest("`flash_attention` testing requires the Jax backend.")

        with patch("keras.src.backend.nn.dot_product_attention") as mock_func:
            causal_lm = Gemma3CausalLM(**self.text_init_kwargs)
            causal_lm.generate("the quick brown fox")
            if running_on_gpu():
                mock_func.assert_called()
            else:
                mock_func.assert_not_called()

    def test_text_early_stopping(self):
        causal_lm = Gemma3CausalLM(**self.text_init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.text_preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_text_multitoken_stopping(self):
        causal_lm = Gemma3CausalLM(**self.text_init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.text_preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]

            output = causal_lm.generate(prompt, stop_token_ids=(3,))
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_text_generate_compilation(self):
        causal_lm = Gemma3CausalLM(**self.text_init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.kaggle_key_required
    @pytest.mark.large
    def test_text_saved_model(self):
        self.run_model_saving_test(
            cls=Gemma3CausalLM,
            init_kwargs=self.text_init_kwargs,
            input_data=self.text_input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma3CausalLM.presets:
            self.run_preset_test(
                cls=Gemma3CausalLM,
                preset=preset,
                input_data=self.text_input_data,
            )
