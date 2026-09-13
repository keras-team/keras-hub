import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_dinov2


class TestDINOV2Converter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = DINOV2Backbone.from_preset(
            "hf://facebook/dinov2-small",
            image_shape=(224, 224, 3),
        )
        dummy_input = {"images": np.ones((1, 224, 224, 3), dtype="float32")}
        output = model.predict(dummy_input)
        self.assertEqual(output.shape[0], 1)

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://facebook/dinov2-small",
            image_shape=(224, 224, 3),
            load_weights=False,
        )
        self.assertIsInstance(model, DINOV2Backbone)

    def test_convert_backbone_config_with_registers(self):
        base_config = {
            "image_size": 224,
            "patch_size": 16,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "mlp_ratio": 4,
            "layerscale_value": 1.0,
            "use_swiglu_ffn": False,
            "hidden_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
        }

        dinov2_config = {**base_config, "model_type": "dinov2"}
        keras_config = convert_dinov2.convert_backbone_config(dinov2_config)
        self.assertFalse(keras_config["antialias_in_interpolation"])
        self.assertEqual(keras_config["num_register_tokens"], 0)

        registers_config = {
            **base_config,
            "model_type": "dinov2_with_registers",
            "num_register_tokens": 4,
        }
        keras_config = convert_dinov2.convert_backbone_config(registers_config)
        self.assertTrue(keras_config["antialias_in_interpolation"])
        self.assertEqual(keras_config["num_register_tokens"], 4)
