import numpy as np
import pytest

from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_image_classifier import DeiTImageClassifier
from keras_hub.src.models.deit.deit_image_classifier_preprocessor import (
    DeiTImageClassifierPreprocessor,
)
from keras_hub.src.models.deit.deit_image_converter import DeiTImageConverter
from keras_hub.src.tests.test_case import TestCase


class DeiTImageClassifierTest(TestCase):
    def setUp(self):
        self.images = np.ones((2, 28, 28, 3))
        self.labels = [0, 1]
        self.backbone = DeiTBackbone(
            image_shape=(28, 28, 3),
            patch_size=4,
            num_layers=3,
            num_heads=6,
            hidden_dim=48,
            intermediate_dim=48 * 4,
        )
        image_converter = DeiTImageConverter(
            image_size=(28, 28),
            scale=1 / 255.0,
        )
        preprocessor = DeiTImageClassifierPreprocessor(
            image_converter=image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "preprocessor": preprocessor,
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        self.run_task_test(
            cls=DeiTImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = DeiTImageClassifier(**self.init_kwargs, head_dtype="bfloat16")
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DeiTImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=DeiTImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
