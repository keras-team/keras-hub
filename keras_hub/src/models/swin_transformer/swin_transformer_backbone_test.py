import pytest
from keras import ops

from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class SwinTransformerBackboneTest(TestCase):
    def setUp(self):
        super().setUp()
        self.init_kwargs = {
            "image_shape": (224, 224, 3),
            "patch_size": 4,
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": 7,
            "mlp_ratio": 4.0,
        }
        self.input_data = ops.ones((1, 224, 224, 3))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 7, 7, 768),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        pass  # Will be added in a future PR when presets are implemented

    @pytest.mark.extra_large
    def test_all_presets(self):
        pass  # Will be added in a future PR when presets are implemented
