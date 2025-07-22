import keras
import pytest

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class LayoutLMv3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "intermediate_dim": 128,
            "max_sequence_length": 128,
            "spatial_embedding_dim": 32,
        }
        self.input_data = {
            "token_ids": keras.ops.ones((2, 10), dtype="int32"),
            "padding_mask": keras.ops.ones((2, 10), dtype="int32"),
            "bbox": keras.ops.ones((2, 10, 4), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 10, 64),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
