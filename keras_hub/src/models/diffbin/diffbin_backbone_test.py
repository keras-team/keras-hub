import keras
import pytest

from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DiffBinBackboneTest(TestCase):
    def setUp(self):
        self.resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 128, 256, 512],
            "stackwise_num_blocks": [3, 4, 6, 3],
            "stackwise_num_strides": [1, 2, 2, 2],
            "block_type": "basic_block",
            "use_pre_activation": False,
            "image_shape": (640, 640, 3),
        }
        policy = keras.mixed_precision.global_policy()
        self.image_encoder = ResNetBackbone(dtype=policy, **self.resnet_kwargs)
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "fpn_channels": 256,
            "head_kernel_list": [3, 2, 2],
        }

        if keras.config.image_data_format() == "channels_first":
            self.input_data = keras.ops.ones(
                shape=(2, 3, 640, 640), dtype="float32"
            )
        else:
            self.input_data = keras.ops.ones(
                shape=(2, 640, 640, 3), dtype="float32"
            )

    def test_diffbin_segmentation_output(self):
        expected_output_shape = {
            "probability_maps": (2, 640, 640, 1),
            "threshold_maps": (2, 640, 640, 1),
        }

        self.run_vision_backbone_test(
            cls=DiffBinBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_output_shape,
            run_mixed_precision_check=True,
            run_quantization_check=False,
            run_data_format_check=True,
        )

    @pytest.mark.large
    def test_save_model(self):
        self.run_model_saving_test(
            cls=DiffBinBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            atol=1e-5,
        )
