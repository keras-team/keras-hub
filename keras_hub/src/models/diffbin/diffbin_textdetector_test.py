import numpy as np
import pytest

from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.diffbin.diffbin_preprocessor import (
    DiffBinPreprocessor,
)
from keras_hub.src.models.diffbin.diffbin_textdetector import (
    DiffBinImageTextDetector,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DiffBinImageTextDetectorTest(TestCase):
    def setUp(self):
        self.images = np.ones((2, 32, 32, 3))
        self.labels = np.concatenate(
            (np.zeros((2, 16, 32, 4)), np.ones((2, 16, 32, 4))), axis=1
        )
        image_encoder = ResNetBackbone(
            input_conv_filters=[4],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 4, 4, 4],
            stackwise_num_blocks=[3, 4, 6, 3],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            image_shape=(32, 32, 3),
        )
        self.backbone = DiffBinBackbone(image_encoder=image_encoder)
        self.preprocessor = DiffBinPreprocessor()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        self.run_task_test(
            cls=DiffBinImageTextDetector,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 32, 32, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DiffBinImageTextDetector,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = DiffBinImageTextDetector(**self.init_kwargs)
        outputs = model.predict(self.images)
        self.assertAllEqual(outputs.shape, (2, 32, 32, 3))

    @pytest.mark.skip(reason="disabled until preset's been uploaded to Kaggle")
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DiffBinImageTextDetector.presets:
            self.run_preset_test(
                cls=DiffBinImageTextDetector,
                preset=preset,
                input_data=self.images,
                expected_output_shape=(2, 32, 32, 3),
            )
