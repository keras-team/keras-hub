import pytest
from keras import ops

from keras_hub.src.models.detr.detr_backbone import DETRBackbone
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DETRBackboneTest(TestCase):
    def setUp(self):
        resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 128, 256],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "block_type": "bottleneck_block",
            "use_pre_activation": False,
        }
        image_encoder = ResNetBackbone(**resnet_kwargs)

        self.init_kwargs = {
            "image_encoder": image_encoder,
            "hidden_dim": 256,
            "num_encoder_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512,
            "dropout": 0.1,
            "activation": "relu",
            "image_shape": (None, None, 3),
        }

        self.input_size = 256
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=DETRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoded_features": (2, 256, 256),
                "pos_embed": (2, 256, 256),
                "mask": (2, 256),
            },
            run_mixed_precision_check=False,
            run_quantization_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DETRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DETRBackbone.presets:
            self.run_preset_test(
                cls=DETRBackbone,
                preset=preset,
                input_data=self.input_data,
            )
