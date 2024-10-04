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
            stackwise_expansion=[1, 4, 6],
            stackwise_num_blocks=[2, 3, 2, 3],
            stackwise_num_filters=[4, 8, 16],
            stackwise_kernel_size=[[3, 3], [5, 5, 5], [5, 5], [5, 5, 5]],
            stackwise_num_strides=[[2, 1], [2, 1, 1], [1, 1], [2, 1, 1]],
            stackwise_se_ratio=[
                [None, None],
                [0.25, 0.25, 0.25],
                [0.3, 0.3],
                [0.3, 0.25, 0.25],
            ],
            stackwise_activation=[
                ["relu", "relu"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish", "hard_swish"],
            ],
            output_num_filters=288,
            input_activation="hard_swish",
            output_activation="hard_swish",
            inverted_res_block=True,
            input_num_filters=16,
            image_shape=(224, 224, 3),
            depthwise_filters=8,
            squeeze_and_excite=0.5,
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
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
