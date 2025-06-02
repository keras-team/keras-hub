import pytest
from keras import ops

from keras_hub.src.models.swin_transformers.swin_transformers_backbone import (
    SwinTransformersBackbone,
)
from keras_hub.src.tests.test_case import TestCase

class SwinTransformersBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "image_shape": (64, 64, 3),
            "patch_size": 2,
            "embed_dim": 32,
            "depths": [1, 1, 1, 1],
            "num_heads": [1, 2, 4, 8],
            "window_size": 4,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "dropout_rate": 0.0,
            "attention_dropout": 0.0,
            "path_dropout": 0.1,
            "patch_norm": True,
            "data_format": "channels_last",
            "dtype": "float32",
        }
        self.input_data = ops.ones((2, 64, 64, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SwinTransformersBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 2, 2, 256),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SwinTransformersBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        pass  # Will be added in a future PR when presets are implemented

    @pytest.mark.extra_large
    def test_all_presets(self):
        pass  # Will be added in a future PR when presets are implemented
