import os
from unittest.mock import patch
import pytest
from keras import ops

from keras_hub.src.models.deepseek_v31.deepseek_v31_backbone import (
    DeepSeekV31Backbone,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_causal_lm import (
    DeepSeekV31CausalLM,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_causal_lm_preprocessor import (
    DeepSeekV31CausalLMPreprocessor,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_tokenizer import (
    DeepSeekV31Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV31CausalLMTest(TestCase):
    def setUp(self):
        # Explicit special tokens for default ID injection
        # Use same vocab as preprocessor test but with larger overall vocabulary
        self.vocab = {
            "<｜begin▁of▁sentence｜>": 151646,
            "<｜end▁of▁sentence｜>": 151643,
            "a": 2,
            "b": 3,
            "c": 4,
            "d": 5,
            "Ġ": 6,
            " ": 7,
        }
        self.merges = []
        self.tokenizer = DeepSeekV31Tokenizer(
            vocabulary=self.vocab, merges=self.merges
        )

        self.preprocessor = DeepSeekV31CausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )

        # Use large vocabulary size to match expected output shape
        self.backbone = DeepSeekV31Backbone(
            vocabulary_size=151650,
            num_layers=2,
            hidden_dim=32,
            num_query_heads=4,
            num_key_value_heads=4,
            intermediate_dim=64,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            num_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["a b", "b a"],)

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=DeepSeekV31CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 151650),
        )

    def test_generate(self):
        causal_lm = DeepSeekV31CausalLM(**self.init_kwargs)
        prompt = "a b"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV31CausalLM.presets:
            self.run_preset_test(
                cls=DeepSeekV31CausalLM,
                preset=preset,
                input_data=self.train_data,
            )
