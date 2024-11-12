import pytest
from keras import ops

from keras_hub.src.models.basnet.basnet import BASNet
from keras_hub.src.models.basnet.basnet_preprocessor import BASNetPreprocessor
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class BASNetTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 64, 64, 3))
        self.backbone = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[2, 2, 2, 2],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="basic_block",
        )
        self.preprocessor = BASNetPreprocessor()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": 1,
        }

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BASNet,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = BASNet(**self.init_kwargs)
        outputs = model.predict(self.images)
        self.assertAllEqual(
            [output.shape for output in outputs], [(2, 64, 64, 1)] * 8
        )
