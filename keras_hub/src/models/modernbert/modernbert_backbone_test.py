import pytest
from keras import ops

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 8,
            "intermediate_dim": 32,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
