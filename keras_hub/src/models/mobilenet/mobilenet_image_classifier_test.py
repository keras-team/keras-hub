import numpy as np
import pytest

from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_classifier import (
    MobileNetImageClassifier,
)
from keras_hub.src.models.mobilenet.mobilenet_image_classifier_preprocessor import (  # noqa: E501
    MobileNetImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenet.mobilenet_image_converter import (
    MobileNetImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MobileNetImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 32, 32, 3), dtype="float32")
        self.labels = [0, 2]
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
            image_shape=(32, 32, 3),
            depthwise_filters=8,
            depthwise_stride=2,
            depthwise_residual=False,
            squeeze_and_excite=0.5,
            last_layer_filter=288,
        )
        self.preprocessor = MobileNetImageClassifierPreprocessor()
        self.image_converter = MobileNetImageConverter(
            height=32, width=32, scale=1 / 255.0
        )
        self.preprocessor = MobileNetImageClassifierPreprocessor(
            self.image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": 3,
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        self.run_task_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 3),
        )

    @pytest.mark.large
    def test_all_presets(self):
        for preset in MobileNetImageClassifier.presets:
            self.run_preset_test(
                cls=MobileNetImageClassifier,
                preset=preset,
                input_data=self.images,
                expected_output_shape=(2, 1000),
            )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
