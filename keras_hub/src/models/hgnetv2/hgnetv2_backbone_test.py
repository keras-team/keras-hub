import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.tests.test_case import TestCase


class HGNetV2BackboneTest(TestCase):
    def setUp(self):
        self.default_input_shape = (64, 64, 3)
        self.num_channels = self.default_input_shape[-1]
        self.stem_channels = [self.num_channels, 16, 32]
        self.default_stage_in_channels = [self.stem_channels[-1], 64]
        self.default_stage_mid_channels = [16, 32]
        self.default_stage_out_channels = [64, 128]
        self.default_num_stages = len(self.default_stage_in_channels)

        self.init_kwargs = {
            "initializer_range": 0.02,
            "depths": [1] * self.default_num_stages,
            "embedding_size": self.stem_channels[-1],
            "hidden_sizes": self.default_stage_out_channels,
            "stem_channels": self.stem_channels,
            "hidden_act": "relu",
            "use_learnable_affine_block": False,
            "num_channels": self.num_channels,
            "stage_in_channels": self.default_stage_in_channels,
            "stage_mid_channels": self.default_stage_mid_channels,
            "stage_out_channels": self.default_stage_out_channels,
            "stage_num_blocks": [1] * self.default_num_stages,
            "stage_numb_of_layers": [1] * self.default_num_stages,
            "stage_downsample": [False, True],
            "stage_light_block": [False, False],
            "stage_kernel_size": [3] * self.default_num_stages,
            "image_shape": self.default_input_shape,
        }
        self.input_size = self.default_input_shape[:2]
        self.batch_size = 2
        self.input_data = keras.ops.convert_to_tensor(
            np.random.rand(self.batch_size, *self.default_input_shape).astype(
                np.float32
            )
        )

    @parameterized.named_parameters(
        (
            "default_config",
            [False, True],
            [False, False],
            2,
            {"stage0": (2, 16, 16, 64), "stage1": (2, 8, 8, 128)},
        ),
        (
            "early_downsample_light_blocks",
            [True, True],
            [True, True],
            2,
            {"stage0": (2, 8, 8, 64), "stage1": (2, 4, 4, 128)},
        ),
        (
            "single_stage_no_downsample",
            [False],
            [False],
            1,
            {"stage0": (2, 16, 16, 64)},
        ),
        (
            "all_no_downsample",
            [False, False],
            [False, False],
            2,
            {"stage0": (2, 16, 16, 64), "stage1": (2, 16, 16, 128)},
        ),
    )
    def test_backbone_basics(
        self,
        stage_downsample_config,
        stage_light_block_config,
        num_stages,
        expected_shapes,
    ):
        current_init_kwargs = self.init_kwargs.copy()
        current_init_kwargs["depths"] = [1] * num_stages
        current_init_kwargs["hidden_sizes"] = self.default_stage_out_channels[
            :num_stages
        ]
        current_init_kwargs["stage_in_channels"] = (
            self.default_stage_in_channels[:num_stages]
        )
        current_init_kwargs["stage_mid_channels"] = (
            self.default_stage_mid_channels[:num_stages]
        )
        current_init_kwargs["stage_out_channels"] = (
            self.default_stage_out_channels[:num_stages]
        )
        current_init_kwargs["stage_num_blocks"] = [1] * num_stages
        current_init_kwargs["stage_numb_of_layers"] = [1] * num_stages
        current_init_kwargs["stage_kernel_size"] = [3] * num_stages
        current_init_kwargs["stage_downsample"] = stage_downsample_config
        current_init_kwargs["stage_light_block"] = stage_light_block_config
        if num_stages > 0:
            current_init_kwargs["stage_in_channels"][0] = self.stem_channels[-1]
            for i in range(1, num_stages):
                current_init_kwargs["stage_in_channels"][i] = (
                    current_init_kwargs["stage_out_channels"][i - 1]
                )
        self.run_vision_backbone_test(
            cls=HGNetV2Backbone,
            init_kwargs=current_init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_shapes,
            run_mixed_precision_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=HGNetV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in HGNetV2Backbone.presets:
            self.run_preset_test(
                cls=HGNetV2Backbone,
                preset=preset,
                input_data=self.input_data,
            )
