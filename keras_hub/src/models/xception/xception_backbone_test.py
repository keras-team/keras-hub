import pytest
from keras import ops

from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.tests.test_case import TestCase


class XceptionBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_conv_filters": [[32, 64], [128, 128], [256, 256]],
            "stackwise_pooling": [False, True, False],
            "image_shape": (None, None, 3),
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=XceptionBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 15, 15, 256),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=XceptionBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XceptionBackbone.presets:
            self.run_preset_test(
                cls=XceptionBackbone,
                preset=preset,
                input_data=self.input_data,
            )
