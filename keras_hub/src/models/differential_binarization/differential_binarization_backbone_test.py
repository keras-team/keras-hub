from keras import ops

from keras_hub.src.models.differential_binarization.differential_binarization_backbone import (
    DifferentialBinarizationBackbone,
)
from keras_hub.src.models.differential_binarization.differential_binarization_preprocessor import (
    DifferentialBinarizationPreprocessor,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DifferentialBinarizationTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 16
        self.images = ops.ones((2, 224, 224, 3))
        self.image_encoder = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[3, 4, 6, 3],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            image_shape=(224, 224, 3),
        )
        self.preprocessor = DifferentialBinarizationPreprocessor()
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DifferentialBinarizationBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            expected_output_shape=(
                2,
                56,
                56,
                256,
            ),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )
