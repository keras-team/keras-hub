import keras
import pytest

from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_presets import (
    backbone_presets,
)
from keras_hub.src.tests.test_case import TestCase


class MobileNetV5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = backbone_presets["mobilenetv5_base"]["config"]

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=keras.ops.ones((2, 256, 256, 3)),
            expected_output_shape=(2, 16, 16, 2048),
            # run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetV5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=keras.ops.ones((2, 256, 256, 3)),
        )
