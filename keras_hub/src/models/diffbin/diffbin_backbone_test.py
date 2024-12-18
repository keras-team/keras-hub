from keras import ops

from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.diffbin.diffbin_preprocessor import (
    DiffBinPreprocessor,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DiffBinTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 32, 32, 3))
        self.image_encoder = ResNetBackbone(
            input_conv_filters=[4],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 4, 4, 4],
            stackwise_num_blocks=[3, 4, 6, 3],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            image_shape=(32, 32, 3),
        )
        self.preprocessor = DiffBinPreprocessor()
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "fpn_channels": 16,
            "head_kernel_list": [3, 2, 2],
        }

    def test_backbone_basics(self):
        expected_output_shape = {
            "probability_maps": (2, 32, 32, 1),
            "threshold_maps": (2, 32, 32, 1),
        }
        self.run_backbone_test(
            cls=DiffBinBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            expected_output_shape=expected_output_shape,
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )
