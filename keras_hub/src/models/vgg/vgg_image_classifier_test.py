import numpy as np
import pytest

from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_image_classifier import VGGImageClassifier
from keras_hub.src.tests.test_case import TestCase


class VGGImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 4, 4, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = VGGBackbone(
            stackwise_num_repeats=[2, 4, 4],
            stackwise_num_filters=[2, 16, 16],
            image_shape=(4, 4, 3),
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
            "pooling": "max",
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
