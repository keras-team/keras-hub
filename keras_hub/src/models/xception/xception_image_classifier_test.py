import numpy as np
import pytest

from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.models.xception.xception_image_classifier import (
    XceptionImageClassifier,
)
from keras_hub.src.models.xception.xception_image_classifier_preprocessor import (  # noqa: E501
    XceptionImageClassifierPreprocessor,
)
from keras_hub.src.models.xception.xception_image_converter import (
    XceptionImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class XceptionImageClassifierTest(TestCase):
    def setUp(self):
        self.images = np.ones((2, 299, 299, 3))
        self.labels = [0, 1]
        self.backbone = XceptionBackbone(
            stackwise_conv_filters=[[32, 64], [128, 128], [256, 256]],
            stackwise_pooling=[False, True, False],
        )
        self.image_converter = XceptionImageConverter(
            image_size=(299, 299),
            scale=1.0 / 127.5,
            offset=-1.0,
        )
        self.preprocessor = XceptionImageClassifierPreprocessor(
            image_converter=self.image_converter,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        self.run_task_test(
            cls=XceptionImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = XceptionImageClassifier(
            **self.init_kwargs, head_dtype="bfloat16"
        )
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...].astype("float32")
        image_batch = self.image_converter(image_batch)
        self.run_preset_test(
            cls=XceptionImageClassifier,
            preset="xception_41_imagenet",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=XceptionImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=XceptionImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XceptionImageClassifier.presets:
            self.run_preset_test(
                cls=XceptionImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.images,
                expected_output_shape=(2, 2),
            )
