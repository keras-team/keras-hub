import pytest
from keras import ops

from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.tests.test_case import TestCase


class FalconBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_attention_heads": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=FalconBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=FalconBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
