import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone
from keras_hub.src.tests.test_case import TestCase


class CSPNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stem_filters": 32,
            "stem_kernel_size": 3,
            "stem_strides": 1,
            "stackwise_strides": 2,
            "stackwise_depth": [1, 2, 8],
            "stackwise_num_filters": [16, 24, 48],
            "image_shape": (None, None, 3),
            "down_growth": True,
            "bottle_ratio": (0.5,) + (1.0,),
            "block_ratio": (1.0,) + (0.5,),
            "expand_ratio": (2.0,) + (1.0,),
            "block_type": "dark_block",
            "stage_type": "csp",
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    @parameterized.named_parameters(
        ("cspnet", "csp", "dark_block"),
    )
    def test_backbone_basics(self, stage_type, block_type):
        self.run_vision_backbone_test(
            cls=CSPNetBackbone,
            init_kwargs={
                **self.init_kwargs,
                "block_type": block_type,
                "stage_type": stage_type,
            },
            input_data=self.input_data,
            expected_output_shape=(2, 6, 6, 48),
            expected_pyramid_output_keys=["P2", "P3", "P4"],
            expected_pyramid_image_sizes=[(30, 30), (14, 14), (6, 6)],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=CSPNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in CSPNetBackbone.presets:
            self.run_preset_test(
                cls=CSPNetBackbone,
                preset=preset,
                input_data=self.input_data,
            )
