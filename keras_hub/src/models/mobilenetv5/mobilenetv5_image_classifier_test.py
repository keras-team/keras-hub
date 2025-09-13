import numpy as np
import pytest

from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import decode_arch_def
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier import (
    MobileNetV5ImageClassifier,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier_preprocessor import (  # noqa: E501
    MobileNetV5ImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_converter import (
    MobileNetV5ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MobileNetV5ImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 32, 32, 3), dtype="float32")
        self.labels = [0, 9]  # num_classes = 10
        arch_def = [
            ["er_r1_k3_s2_e4_c24"],
            ["uir_r2_k5_s2_e6_c48"],
        ]
        block_args = decode_arch_def(arch_def)
        self.backbone = MobileNetV5Backbone(
            block_args=block_args,
            input_shape=(32, 32, 3),
            stem_size=16,
            use_msfa=False,
        )
        self.image_converter = MobileNetV5ImageConverter(
            height=32, width=32, scale=1 / 255.0
        )
        self.preprocessor = MobileNetV5ImageClassifierPreprocessor(
            self.image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": 10,
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        self.run_task_test(
            cls=MobileNetV5ImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 10),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetV5ImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
