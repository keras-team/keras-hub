import pytest
from keras import ops

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone
from keras_hub.src.tests.test_case import TestCase


class LayoutLMv3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "num_layers": 2,
            "num_heads": 2,
            "intermediate_dim": 16,
            "max_sequence_length": 5,
            "max_spatial_positions": 10,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "bbox": ops.zeros((2, 5, 4), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
