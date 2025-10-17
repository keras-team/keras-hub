import keras
import pytest

from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class MobileNetV5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_block_types": [["er"], ["uir", "uir"]],
            "stackwise_num_blocks": [1, 2],
            "stackwise_num_filters": [[24], [48, 48]],
            "stackwise_strides": [[2], [2, 1]],
            "stackwise_act_layers": [["relu"], ["relu", "relu"]],
            "stackwise_exp_ratios": [[4.0], [6.0, 6.0]],
            "stackwise_se_ratios": [[0.0], [0.0, 0.0]],
            "stackwise_dw_kernel_sizes": [[0], [5, 5]],
            "stackwise_dw_start_kernel_sizes": [[0], [0, 0]],
            "stackwise_dw_end_kernel_sizes": [[0], [0, 0]],
            "stackwise_exp_kernel_sizes": [[3], [0, 0]],
            "stackwise_pw_kernel_sizes": [[1], [0, 0]],
            "stackwise_num_heads": [[0], [0, 0]],
            "stackwise_key_dims": [[0], [0, 0]],
            "stackwise_value_dims": [[0], [0, 0]],
            "stackwise_kv_strides": [[0], [0, 0]],
            "stackwise_use_cpe": [[False], [False, False]],
            "image_shape": (32, 32, 3),
            "stem_size": 16,
            "use_msfa": False,
        }
        self.input_data = keras.ops.ones((2, 32, 32, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                2,
                4,
                4,
                48,
            ),
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MobileNetV5Backbone,
            preset="mobilenetv5_300m_enc.gemma3n",
            input_data=keras.ops.ones((1, 224, 224, 3)),
            expected_output_shape=(1, 16, 16, 2048),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
