import numpy as np
import pytest

from keras_hub.src.models.basnet.basnet import BASNetImageSegmenter
from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.basnet.basnet_image_converter import (
    BASNetImageConverter,
)
from keras_hub.src.models.basnet.basnet_preprocessor import BASNetPreprocessor
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class BASNetTest(TestCase):
    def setUp(self):
        self.images = np.ones((2, 64, 64, 3))
        self.labels = np.concatenate(
            (np.zeros((2, 32, 64, 1)), np.ones((2, 32, 64, 1))), axis=1
        )
        self.image_encoder = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[2, 2, 2, 2],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="basic_block",
        )
        self.backbone = BASNetBackbone(
            image_encoder=self.image_encoder,
            num_classes=1,
        )
        self.preprocessor = BASNetPreprocessor(
            image_converter=BASNetImageConverter(height=64, width=64)
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        self.run_task_test(
            cls=BASNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 64, 64, 1),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BASNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.skip(reason="TODO: Bug with BASNet liteRT export")
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=BASNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = BASNetImageSegmenter(**self.init_kwargs)
        output = model.predict(self.images)
        self.assertAllEqual(output.shape, (2, 64, 64, 1))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BASNetImageSegmenter.presets:
            self.run_preset_test(
                cls=BASNetImageSegmenter,
                preset=preset,
                input_data=self.images,
                expected_output_shape=(2, 64, 64, 1),
            )
