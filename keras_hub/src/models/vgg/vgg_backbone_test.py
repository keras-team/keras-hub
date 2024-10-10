import numpy as np
import pytest

from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.tests.test_case import TestCase


class VGGBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_num_repeats": [2, 3, 3],
            "stackwise_num_filters": [8, 64, 64],
            "image_shape": (16, 16, 3),
        }
        self.input_data = np.ones((2, 16, 16, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=VGGBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 2, 2, 64),
            run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VGGBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
