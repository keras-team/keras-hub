import numpy as np
import pytest

from keras_hub.src.models.yolo_v8.yolo_v8_backbone import YOLOV8Backbone
from keras_hub.src.tests.test_case import TestCase


class YOLOV8BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_channels": [64, 128, 256, 512],
            "stackwise_depth": [1, 2, 2, 1],
            "activation": "swish",
            "image_shape": (32, 32, 3),
        }
        self.input_size = 32
        self.input_data = np.ones(
            (2, self.input_size, self.input_size, 3), dtype="float32"
        )

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=YOLOV8Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 1, 1, 512),
            expected_pyramid_output_keys=["P1", "P2", "P3", "P4", "P5"],
            expected_pyramid_image_sizes=[
                (8, 8),
                (8, 8),
                (4, 4),
                (2, 2),
                (1, 1),
            ],
            run_mixed_precision_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=YOLOV8Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
