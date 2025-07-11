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
        self.batch_size = 2
        self.stem_channels = [self.num_channels, 16, 32]
        self.default_stage_out_filters = [64, 128]
        self.default_num_stages = 2
        self.stackwise_stage_filters = [
            [32, 16, 64, 1, 1, 3],
            [64, 32, 128, 1, 1, 3],
        ]
        self.init_kwargs = {
            "embedding_size": self.stem_channels[-1],
            "stem_channels": self.stem_channels,
            "hidden_act": "relu",
            "use_learnable_affine_block": False,
            "image_shape": self.default_input_shape,
            "depths": [1] * self.default_num_stages,
            "hidden_sizes": [
                stage[2] for stage in self.stackwise_stage_filters
            ],
            "stackwise_stage_filters": self.stackwise_stage_filters,
            "apply_downsample": [False, True],
            "use_lightweight_conv_block": [False, False],
            # Explicitly pass the out_features arg to ensure comprehensive
            # test coverage for D-FINE.
            "out_features": ["stem", "stage1", "stage2"],
        }
        self.input_data = keras.ops.convert_to_tensor(
            np.random.rand(self.batch_size, *self.default_input_shape).astype(
                np.float32
            )
        )

    @parameterized.named_parameters(
        (
            "default",
            [False, True],
            [False, False],
            2,
            {
                "stem": (2, 16, 16, 32),
                "stage1": (2, 16, 16, 64),
                "stage2": (2, 8, 8, 128),
            },
        ),
        (
            "early_downsample_light_blocks",
            [True, True],
            [True, True],
            2,
            {
                "stem": (2, 16, 16, 32),
                "stage1": (2, 8, 8, 64),
                "stage2": (2, 4, 4, 128),
            },
        ),
        (
            "single_stage_no_downsample",
            [False],
            [False],
            1,
            {
                "stem": (2, 16, 16, 32),
                "stage1": (2, 16, 16, 64),
            },
        ),
        (
            "all_no_downsample",
            [False, False],
            [False, False],
            2,
            {
                "stem": (2, 16, 16, 32),
                "stage1": (2, 16, 16, 64),
                "stage2": (2, 16, 16, 128),
            },
        ),
    )
    def test_backbone_basics(
        self,
        apply_downsample,
        use_lightweight_conv_block,
        num_stages,
        expected_shapes,
    ):
        test_filters = self.stackwise_stage_filters[:num_stages]
        hidden_sizes = [stage[2] for stage in test_filters]
        test_kwargs = {
            **self.init_kwargs,
            "depths": [1] * num_stages,
            "hidden_sizes": hidden_sizes,
            "stackwise_stage_filters": test_filters,
            "apply_downsample": apply_downsample,
            "use_lightweight_conv_block": use_lightweight_conv_block,
            "out_features": ["stem"]
            + [f"stage{i + 1}" for i in range(num_stages)],
        }
        self.run_vision_backbone_test(
            cls=HGNetV2Backbone,
            init_kwargs=test_kwargs,
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
