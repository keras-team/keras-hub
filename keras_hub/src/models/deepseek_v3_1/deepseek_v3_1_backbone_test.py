import pytest
from keras import ops

from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_backbone import (
    DeepSeekV3_1Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV3_1BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "hidden_dim": 64,
            "num_query_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_dim": 128,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "num_routed_experts": 4,
            "num_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="bool"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DeepSeekV3_1Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 64),
        )

    def test_num_parameters(self):
        model = DeepSeekV3_1Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_backbone_with_cache(self):
        model = DeepSeekV3_1Backbone(**self.init_kwargs)
        cache = model._build_cache(batch_size=2, sequence_length=5)
        self.assertIsInstance(cache, list)
        self.assertEqual(len(cache), self.init_kwargs["num_layers"])

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV3_1Backbone.presets:
            self.run_preset_test(
                cls=DeepSeekV3_1Backbone,
                preset=preset,
                input_data=self.input_data,
            )
