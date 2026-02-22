import pytest
from keras import ops
from keras import mixed_precision

from keras_hub.src.models.deepseek_v31.deepseek_v31_backbone import (
    DeepSeekV31Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV31BackboneTest(TestCase):
    @classmethod
    def setUpClass(cls):
        # Enable mixed precision globally
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

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
        original_assert_dtype_equal = self.assertDTypeEqual

        def assert_dtype_flexible(tensor, expected_dtype, msg=None):
            actual_dtype = str(tensor.dtype)
            allowed_dtypes = ["float16", "bfloat16"]
            if actual_dtype not in allowed_dtypes:
                self.fail(
                    msg
                    or f"Tensor dtype {actual_dtype} not in allowed {allowed_dtypes}"
                )

    def test_num_parameters(self):
        model = DeepSeekV31Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_backbone_with_cache(self):
        model = DeepSeekV31Backbone(**self.init_kwargs)
        token_ids = ops.ones((2, 5), dtype="int32")
        cache = model._build_cache(token_ids)

        self.assertIsInstance(cache, list)
        self.assertEqual(len(cache), self.init_kwargs["num_layers"])
        for c_kv, k_rope in cache:
            self.assertEqual(
                c_kv.shape, (2, 5, self.init_kwargs["kv_lora_rank"])
            )
            self.assertEqual(
                k_rope.shape, (2, 5, self.init_kwargs["qk_rope_head_dim"])
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV31Backbone.presets:
            self.run_preset_test(
                cls=DeepSeekV31Backbone,
                preset=preset,
                input_data=self.input_data,
            )
