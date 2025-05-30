import keras
import numpy as np
import pytest

from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase

class DiffBinBackboneTest(TestCase):
    def setUp(self):
        self.resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "block_type": "basic_block",
            "use_pre_activation": False,
        }
        self.image_encoder = ResNetBackbone(**self.resnet_kwargs)
        self.init_kwargs= {
            "image_encoder": self.image_encoder,
            "fpn_channels": 256,
            "head_kernel_list": [3, 2, 2],
        }
        self.input_data= np.ones(shape=(2, 640,640, 3),dtype="float32")

    def test_diffbin_segmentation_output(self):
        self.run_vision_backbone_test(
            cls= DiffBinBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "probability_maps": (2, 640, 640, 1),
                "threshold_maps": (2, 640, 640, 1),
            }, 
            run_mixed_precision_check=False,
            run_quantization_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_save_model(self):
        self.run_model_saving_test(
            cls= DiffBinBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            atol= 1e-5
        )
