import numpy as np
import pytest

from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_classifier import (
    MobileNetImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class MobileNetImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 224, 224, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = MobileNetBackbone(
            stackwise_expansion=[
                [40, 56],
                [64, 144, 144],
                [72, 72],
                [144, 288, 288],
            ],
            stackwise_num_blocks=[2, 3, 2, 3],
            stackwise_num_filters=[
                [16, 16],
                [24, 24, 24],
                [24, 24],
                [48, 48, 48],
            ],
            stackwise_kernel_size=[[3, 3], [5, 5, 5], [5, 5], [5, 5, 5], [1]],
            stackwise_num_strides=[[2, 1], [2, 1, 1], [1, 1], [2, 1, 1], [1]],
            stackwise_se_ratio=[
                [None, None],
                [0.25, 0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            stackwise_activation=[
                ["relu", "relu"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish", "hard_swish"],
            ],
            stackwise_padding=[[1, 1], [2, 2, 2], [2, 2], [2, 2, 2], [1]],
            output_num_filters=1024,
            input_activation="hard_swish",
            output_activation="hard_swish",
            input_num_filters=16,
            image_shape=(224, 224, 3),
            depthwise_filters=8,
            squeeze_and_excite=0.5,
            last_layer_filter=288,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...] / 255.0
        self.run_preset_test(
            cls=MobileNetImageClassifier,
            preset="mobilenetv3_small_050",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
