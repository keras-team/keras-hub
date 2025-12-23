import numpy as np
import pytest

from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_image_classifier import VGGImageClassifier
from keras_hub.src.models.vgg.vgg_image_classifier_preprocessor import (
    VGGImageClassifierPreprocessor,
)
from keras_hub.src.models.vgg.vgg_image_converter import VGGImageConverter
from keras_hub.src.tests.test_case import TestCase


class VGGImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 8, 8, 3), dtype="float32")
        self.labels = [0, 1]
        self.backbone = VGGBackbone(
            stackwise_num_repeats=[2, 4, 4],
            stackwise_num_filters=[2, 16, 16],
            image_shape=(8, 8, 3),
        )
        image_converter = VGGImageConverter(image_size=(8, 8))
        self.preprocessor = VGGImageClassifierPreprocessor(
            image_converter=image_converter,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
            "pooling": "flatten",
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        self.run_task_test(
            cls=VGGImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VGGImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.skip(reason="TODO: Bug with VGGImageClassifier liteRT export")
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=VGGImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        # we need at least 32x32 image resolution here to satisfy the presets'
        # downsampling constraints
        images = np.ones((2, 32, 32, 3), dtype="float32")
        for preset in VGGImageClassifier.presets:
            self.run_preset_test(
                cls=VGGImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=images,
                expected_output_shape=(2, 2),
            )
