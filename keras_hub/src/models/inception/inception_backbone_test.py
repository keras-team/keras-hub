import numpy as np
import pytest

from keras_hub.src.models.inception.inception_backbone import InceptionBackbone
from keras_hub.src.tests.test_case import TestCase


class InceptionBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "initial_filters": [64],
            "initial_strides": [2],
            "inception_config": [
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            "aux_classifiers": False,
            "image_shape": (32, 32, 3),
        }
        self.input_size = 32
        self.input_data = np.ones(
            (2, self.input_size, self.input_size, 3), dtype="float32"
        )

    def test_backbone_basics(self):
        init_kwargs = {
            "initial_filters": [64],
            "initial_strides": [2],
            "inception_config": [
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            "aux_classifiers": False,
            "image_shape": (32, 32, 3),
        }
        self.run_vision_backbone_test(
            cls=InceptionBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 1, 1, 512),
            expected_pyramid_output_keys=["P2", "P3", "P4", "P5"],
            expected_pyramid_image_sizes=[(8, 8), (4, 4), (2, 2), (1, 1)],
            run_mixed_precision_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        init_kwargs = {
            "initial_filters": [64],
            "initial_strides": [2],
            "inception_config": [
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            "aux_classifiers": False,
            "image_shape": (32, 32, 3),
            "dtype": "float32",  
        }
        self.run_model_saving_test(
            cls=InceptionBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )

@pytest.mark.parametrize(
    "aux_classifiers", [True, False]
)
def test_auxiliary_branches(aux_classifiers):
    self = InceptionBackboneTest()
    self.setUp()
    kwargs = {
            "initial_filters": [64],
            "initial_strides": [2],
            "inception_config": [
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            "aux_classifiers": aux_classifiers,
            "image_shape": (32, 32, 3),
            "dtype": "float32",  
        }
        
    backbone = InceptionBackbone(**kwargs)
    outputs = backbone(self.input_data)
        
    if aux_classifiers:
        self.assertIsInstance(outputs, dict)
        self.assertIn("aux1", outputs)
        self.assertIn("aux2", outputs)
        self.assertIn("main", outputs)
    else:
        # When not using auxiliary branches, output should be single tensor
        # or the feature pyramid if enabled
        if (
            isinstance(outputs, dict) 
            and "P2" in outputs
        ):
            self.assertNotIn("aux1", outputs)
            self.assertNotIn("aux2", outputs)