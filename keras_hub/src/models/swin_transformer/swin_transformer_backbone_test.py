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
            "image_shape": (32, 32, 3),
            "embed_dim": 32,
            "depths": (2, 2),
            "num_heads": (2, 4),
            "window_size": 4,
        }
        self.input_data = ops.ones((1, 32, 32, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 16, 64),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

