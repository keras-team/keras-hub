import keras
import pytest
from keras import ops
from packaging import version

from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.tests.test_case import TestCase


class ESMBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 1,
            "hidden_dim": 2,
            "intermediate_dim": 4,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        if version.parse(keras.__version__) < version.parse("3.6"):
            self.skipTest("Failing on keras lower version")
        self.run_backbone_test(
            cls=ESMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ESMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ESMBackbone,
            preset="hf://facebook/esm2_t6_8M_UR50D",
            input_data={
                "token_ids": ops.array([[2, 3, 4, 5]], dtype="int32"),
            },
            expected_output_shape=(1, 4, 320),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [0.081905, -0.245397, 0.324738, 0.27153, -0.006534]
            ),
        )
