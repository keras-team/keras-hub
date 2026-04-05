import numpy as np
import pytest

from keras_hub.src.models.cnn14_panns.cnn14_panns_backbone import (
    Cnn14PannsBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class Cnn14PannsBackboneTest(TestCase):
    def setUp(self):
        # We test with a smaller model and input for performance
        self.init_kwargs = {
            "stackwise_num_filters": [64, 128, 256],
            "input_shape": (32, 32, 1),
        }
        self.input_data = np.ones((2, 32, 32, 1), dtype="float32")

    def test_backbone_basics(self):
        # In CNN14, the first 5 blocks have (2,2) pooling,
        # and the last one has (1,1).
        # With 3 blocks: (32,32) -> (16,16) -> (8,8) (last block is 1x1 pool)
        self.run_backbone_test(
            cls=Cnn14PannsBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 8, 8, 256),
            run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Cnn14PannsBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_variable_input_shape(self):
        # Ensure it works with dynamic time dimensions.
        # Note: frequency bins must be fixed (e.g., 64) for BN on that axis.
        backbone = Cnn14PannsBackbone(
            stackwise_num_filters=[64],
            input_shape=(None, 64, 1),
        )
        input_data = np.ones((2, 128, 64, 1), dtype="float32")
        # 1 block with (1,1) pool (since it's the last block)
        output = backbone(input_data)
        self.assertEqual(output.shape, (2, 128, 64, 64))

    def test_full_cnn14_shape(self):
        # Verify the standard CNN14 architecture output shape
        # with standard AudioSet log-mel input (10s @ 100fps, 64 bins)
        backbone = Cnn14PannsBackbone()
        input_data = np.ones((2, 1000, 64, 1), dtype="float32")
        # (1000, 64) -> (500, 32) -> (250, 16) -> (125, 8) -> (62, 4)
        # -> (31, 2) -> (31, 2)
        # Note: (1,1) pooling on the last block
        output = backbone(input_data)
        self.assertEqual(output.shape, (2, 31, 2, 2048))
