import numpy as np
import pytest

from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone
from keras_hub.src.models.densenet.densenet_image_classifier import (
    DenseNetImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class DenseNetImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 224, 224, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = DenseNetBackbone(
            stackwise_num_repeats=[6, 12, 24, 16],
            compression_ratio=0.5,
            growth_rate=32,
            image_shape=(224, 224, 3),
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
            cls=DenseNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DenseNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
