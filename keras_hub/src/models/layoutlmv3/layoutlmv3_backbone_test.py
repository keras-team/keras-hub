import keras

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class LayoutLMv3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,  # Smaller for testing
            "hidden_dim": 32,        # Smaller for testing
            "num_layers": 1,         # Minimal for testing
            "num_heads": 2,
            "intermediate_dim": 64,
            "max_sequence_length": 16,
            "spatial_embedding_dim": 16,
        }
        self.input_data = {
            "token_ids": keras.ops.ones((1, 4), dtype="int32"),
            "padding_mask": keras.ops.ones((1, 4), dtype="int32"),
            "bbox": keras.ops.ones((1, 4, 4), dtype="int32"),
        }

    def test_backbone_instantiation(self):
        # Test that the model can be created without errors
        model = LayoutLMv3Backbone(**self.init_kwargs)
        self.assertIsNotNone(model)

    def test_backbone_call(self):
        # Test that the model can be called without errors
        model = LayoutLMv3Backbone(**self.init_kwargs)
        output = model(self.input_data)
        # Just check that we get some output
        self.assertIsNotNone(output)
        
    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 4, 32),
        )
