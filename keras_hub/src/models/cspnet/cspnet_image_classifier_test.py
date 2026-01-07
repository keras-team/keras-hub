import numpy as np
import pytest

from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone
from keras_hub.src.models.cspnet.cspnet_image_classifier import (
    CSPNetImageClassifier,
)
from keras_hub.src.models.cspnet.cspnet_image_classifier_preprocessor import (
    CSPNetImageClassifierPreprocessor,
)
from keras_hub.src.models.cspnet.cspnet_image_converter import (
    CSPNetImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class CSPNetImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 32, 32, 3), dtype="float32")
        self.labels = [0, 2]
        self.backbone = CSPNetBackbone(
            stem_filters=32,
            stem_kernel_size=3,
            stem_strides=1,
            stackwise_strides=2,
            stackwise_depth=[1, 2, 8],
            stackwise_num_filters=[16, 24, 48],
            image_shape=(None, None, 3),
            down_growth=True,
            bottle_ratio=(0.5,) + (1.0,),
            block_ratio=(1.0,) + (0.5,),
            expand_ratio=(2.0,) + (1.0,),
            block_type="dark_block",
            stage_type="csp",
        )
        self.image_converter = CSPNetImageConverter(
            height=32, width=32, scale=1 / 255.0
        )
        self.preprocessor = CSPNetImageClassifierPreprocessor(
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
            cls=CSPNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 3),
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        image_batch = self.load_test_image()[None, ...] / 255.0
        self.run_preset_test(
            cls=CSPNetImageClassifier,
            preset="hf://timm/cspdarknet53.ra_in1k",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=CSPNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=CSPNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
