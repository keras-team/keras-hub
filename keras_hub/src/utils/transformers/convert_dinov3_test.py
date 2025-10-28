import numpy as np
import pytest

from keras_hub.src.models.dinov3.dinov3_backbone import DINOV3Backbone
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
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
