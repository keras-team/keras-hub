import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.dinov3.dinov3_backbone import DINOV3Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_dinov3


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        pytest.skip(reason="TODO: enable after HF token is available in CI")
        model = DINOV3Backbone.from_preset(
            "hf://facebook/dinov3-vits16-pretrain-lvd1689m",
            image_shape=(224, 224, 3),
        )
        dummy_input = {
            "pixel_values": np.ones((1, 224, 224, 3), dtype="float32")
        }
        output = model.predict(dummy_input)
        self.assertAllClose(
            output[0, 0, :10],
            [
                -0.2769,
                0.5487,
                0.2501,
                -1.2269,
                0.5886,
                0.0762,
                0.6251,
                0.1874,
                -0.4259,
                -0.4362,
            ],
            atol=1e-2,
        )

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://facebook/dinov3-vits16-pretrain-lvd1689m",
            image_shape=(224, 224, 3),
            load_weights=False,
        )
        self.assertIsInstance(model, DINOV3Backbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "image_size": 224,
            "patch_size": 16,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "intermediate_size": 48,
            "layerscale_value": 1.0,
            "num_register_tokens": 0,
            "hidden_act": "gelu",
            "use_gated_mlp": False,
            "query_bias": True,
            "key_bias": True,
            "value_bias": True,
            "proj_bias": True,
            "mlp_bias": True,
            "attention_dropout": 0.0,
            "drop_path_rate": 0.0,
            "layer_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        keras_config = convert_dinov3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 10000.0)

        # transformers >= 5 format
        transformers_config = {
            "image_size": 224,
            "patch_size": 16,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "intermediate_size": 48,
            "layerscale_value": 1.0,
            "num_register_tokens": 0,
            "hidden_act": "gelu",
            "use_gated_mlp": False,
            "query_bias": True,
            "key_bias": True,
            "value_bias": True,
            "proj_bias": True,
            "mlp_bias": True,
            "attention_dropout": 0.0,
            "drop_path_rate": 0.0,
            "layer_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 20000.0},
        }
        keras_config = convert_dinov3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 20000.0)
