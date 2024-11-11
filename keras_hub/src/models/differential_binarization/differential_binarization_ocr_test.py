import pytest
from keras import ops

from keras_hub.src.models.differential_binarization.differential_binarization_backbone import (
    DifferentialBinarizationBackbone,
)
from keras_hub.src.models.differential_binarization.differential_binarization_ocr import (
    DifferentialBinarizationOCR,
)
from keras_hub.src.models.differential_binarization.differential_binarization_preprocessor import (
    DifferentialBinarizationPreprocessor,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DifferentialBinarizationOCRTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 32, 32, 3))
        self.labels = ops.zeros((2, 32, 32, 4))
        image_encoder = ResNetBackbone(
            input_conv_filters=[4],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 4, 4, 4],
            stackwise_num_blocks=[3, 4, 6, 3],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            image_shape=(32, 32, 3),
        )
        self.backbone = DifferentialBinarizationBackbone(
            image_encoder=image_encoder
        )
        self.preprocessor = DifferentialBinarizationPreprocessor()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        self.run_task_test(
            cls=DifferentialBinarizationOCR,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 32, 32, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DifferentialBinarizationOCR,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = DifferentialBinarizationOCR(**self.init_kwargs)
        outputs = model.predict(self.images)
        self.assertAllEqual(outputs.shape, (2, 32, 32, 3))

    @pytest.mark.skip(reason="disabled until preset's been uploaded to Kaggle")
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DifferentialBinarizationOCR.presets:
            self.run_preset_test(
                cls=DifferentialBinarizationOCR,
                preset=preset,
                input_data=self.images,
                expected_output_shape=(2, 32, 32, 3),
            )
