import keras
import pytest

from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.tests.test_case import TestCase


class MobileNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_expansion": [
                [40, 56],
                [64, 144, 144],
                [72, 72],
                [144, 288, 288],
            ],
            "stackwise_num_blocks": [2, 3, 2, 3],
            "stackwise_num_filters": [
                [16, 16],
                [24, 24, 24],
                [24, 24],
                [48, 48, 48],
            ],
            "stackwise_kernel_size": [
                [3, 3],
                [5, 5, 5],
                [5, 5],
                [5, 5, 5],
            ],
            "stackwise_num_strides": [
                [2, 1],
                [2, 1, 1],
                [1, 1],
                [2, 1, 1],
            ],
            "stackwise_se_ratio": [
                [None, None],
                [0.25, 0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            "stackwise_activation": [
                ["relu", "relu"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish"],
            ],
            "stackwise_padding": [[1, 1], [2, 2, 2], [2, 2], [2, 2, 2]],
            "output_num_filters": 1024,
            "input_activation": "hard_swish",
            "output_activation": "hard_swish",
            "input_num_filters": 16,
            "image_shape": (32, 32, 3),
            "depthwise_filters": 8,
            "depthwise_stride": 2,
            "depthwise_residual": False,
            "squeeze_and_excite": 0.5,
            "last_layer_filter": 288,
        }
        self.input_data = keras.ops.ones((2, 32, 32, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=MobileNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 1, 1, 288),
            run_mixed_precision_check=True,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MobileNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
