from keras import ops

from keras_hub.src.models.roformerV2.roformerV2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class RoformerV2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 1,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
            "head_size": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=RoformerV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )
