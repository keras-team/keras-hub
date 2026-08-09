import numpy as np
import pytest

from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
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
        self.backbone = MobileNetV5Backbone(
            stackwise_block_types=[["er"], ["uir", "uir"]],
            stackwise_num_blocks=[1, 2],
            stackwise_num_filters=[[24], [48, 48]],
            stackwise_strides=[[2], [2, 1]],
            stackwise_act_layers=[["relu"], ["relu", "relu"]],
            stackwise_exp_ratios=[[4.0], [6.0, 6.0]],
            stackwise_se_ratios=[[0.0], [0.0, 0.0]],
            stackwise_dw_kernel_sizes=[[0], [5, 5]],
            stackwise_dw_start_kernel_sizes=[[0], [0, 0]],
            stackwise_dw_end_kernel_sizes=[[0], [0, 0]],
            stackwise_exp_kernel_sizes=[[3], [0, 0]],
            stackwise_pw_kernel_sizes=[[1], [0, 0]],
            stackwise_num_heads=[[0], [0, 0]],
            stackwise_key_dims=[[0], [0, 0]],
            stackwise_value_dims=[[0], [0, 0]],
            stackwise_kv_strides=[[0], [0, 0]],
            stackwise_use_cpe=[[False], [False, False]],
            image_shape=(32, 32, 3),
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

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MobileNetV5ImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )
