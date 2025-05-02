import keras
from keras import ops

from keras_hub.src.models.esm.esm_backbone import (
    ESMBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ESMBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 1,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "head_size": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        if keras.__version__ < "3.6":
            self.skipTest("Failing on keras lower version")
        self.run_backbone_test(
            cls=ESMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )
