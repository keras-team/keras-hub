from keras import ops

from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class BASNetBackboneTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 64, 64, 3))
        self.image_encoder = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[2, 2, 2, 2],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="basic_block",
        )
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "num_classes": 1,
        }

    def test_backbone_basics(self):
        output_names = ["refine_out"] + [
            f"predict_out_{i}" for i in range(1, 8)
        ]
        expected_output_shape = {name: (2, 64, 64, 1) for name in output_names}
        self.run_backbone_test(
            cls=BASNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            expected_output_shape=expected_output_shape,
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )