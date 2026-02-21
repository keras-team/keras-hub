import os
from unittest.mock import patch
import pytest
from keras import ops

from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_backbone import (
    DeepSeekV3_1Backbone,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_causal_lm import (
    DeepSeekV3_1CausalLM,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_causal_lm_preprocessor import (
    DeepSeekV3_1CausalLMPreprocessor,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_tokenizer import (
    DeepSeekV3_1Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV3_1CausalLMTest(TestCase):
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
        self.tokenizer = DeepSeekV3_1Tokenizer(
            vocabulary=self.vocab, merges=self.merges
        )

        self.preprocessor = DeepSeekV3_1CausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )

        # Use large vocabulary size to match expected output shape
        self.backbone = DeepSeekV3_1Backbone(
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
            cls=DeepSeekV3_1CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 151650),
        )

    def test_generate(self):
        causal_lm = DeepSeekV3_1CausalLM(**self.init_kwargs)
        prompt = "a b"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV3_1CausalLM.presets:
            self.run_preset_test(
                cls=DeepSeekV3_1CausalLM,
                preset=preset,
                input_data=self.train_data,
            )
