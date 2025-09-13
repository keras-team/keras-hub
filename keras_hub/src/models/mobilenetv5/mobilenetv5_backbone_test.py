import keras
import pytest

from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import decode_arch_def
from keras_hub.src.tests.test_case import TestCase


class MobileNetV5BackboneTest(TestCase):
    def setUp(self):
        arch_def = [
            ["er_r1_k3_s2_e4_c24"],
            ["uir_r2_k5_s2_e6_c48"],
        ]
        block_args = decode_arch_def(arch_def)

        self.init_kwargs = {
            "block_args": block_args,
            "image_shape": (32, 32, 3),
            "stem_size": 16,
            "use_msfa": False,
        }
        self.input_data = keras.ops.ones((2, 32, 32, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                2,
                4,
                4,
                48,
            ),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
