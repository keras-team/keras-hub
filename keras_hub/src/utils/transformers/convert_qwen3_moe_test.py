import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_causal_lm import Qwen3MoeCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen3_moe


# NOTE: This test is valid and should pass locally. It is skipped only on
# TensorFlow GPU CI because of ResourceExhaustedError (OOM). Revisit once
# TensorFlow GPU CI runs without hitting OOM.
@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3MoeCausalLM.from_preset("hf://Qwen/Qwen3-30B-A3B")
        prompt = "What is the capital of France?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://Qwen/Qwen3-30B-A3B"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3MoeCausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3MoeBackbone)


class TestQwen3MoeRopeTheta(TestCase):
    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "head_dim": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "moe_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "router_aux_loss_coef": 0.001,
        }
        keras_config = convert_qwen3_moe.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_qwen3_moe.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
