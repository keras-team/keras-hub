import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen_moe.qwen_moe_backbone import QwenMoeBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen_moe


class TestQwenMoeConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = QwenMoeBackbone.from_preset(
            "hf://yujiepan/qwen1.5-moe-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, QwenMoeBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen1.5-moe-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, QwenMoeBackbone)

    def test_qwen_moe_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "use_sliding_window": False,
            "sliding_window": 4096,
            "output_router_logits": False,
            "router_aux_loss_coef": 0.001,
        }
        keras_config = convert_qwen_moe.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_qwen_moe.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
