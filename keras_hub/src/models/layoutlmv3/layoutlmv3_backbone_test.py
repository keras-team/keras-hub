import keras
from keras import ops

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
            "token_ids": ops.cast(
                keras.random.uniform(
                    shape=(2, 10), minval=0, maxval=1000, dtype="float32"
                ),
                "int32",
            ),
            "padding_mask": keras.ops.ones((2, 10), dtype="int32"),
            "bbox": ops.cast(
                keras.random.uniform(
                    shape=(2, 10, 4), minval=0, maxval=1000, dtype="float32"
                ),
                "int32",
            ),
        }

    def test_backbone_basics(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        # Call the model to ensure it's built
        _ = model(self.input_data)
        self.assertEqual(model.vocabulary_size, 1000)
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.num_heads, 2)
        self.assertEqual(model.intermediate_dim, 128)
        self.assertEqual(model.max_sequence_length, 128)
        self.assertEqual(model.spatial_embedding_dim, 32)

    def test_backbone_functionality(self):
        """Test backbone using the standardized test helper."""
        self.run_backbone_test(
            cls=LayoutLMv3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 10, 64),
        )

    def test_token_embedding_matrix_property(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        embeddings = model.token_embedding_matrix
        expected_shape = [1000, 64]  # vocabulary_size, hidden_dim
        self.assertEqual(list(embeddings.shape), expected_shape)

    def test_spatial_embeddings_initialization(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)

        # Check that spatial embeddings have correct shapes
        for coord in ["x", "y", "h", "w"]:
            embeddings = model.spatial_embeddings[coord].embeddings
            expected_shape = [1024, 32]  # max_bbox_value, spatial_embedding_dim
            self.assertEqual(list(embeddings.shape), expected_shape)

    def test_bbox_processing(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)

        # Test with bbox values at the boundary
        bbox_data = keras.ops.array(
            [[[0, 0, 100, 50], [100, 100, 200, 150]]], dtype="int32"
        )
        input_data = {
            "token_ids": keras.ops.array([[1, 2]], dtype="int32"),
            "padding_mask": keras.ops.ones((1, 2), dtype="int32"),
            "bbox": bbox_data,
        }

        output = model(input_data)
        expected_shape = [1, 2, 64]
        self.assertEqual(list(output.shape), expected_shape)

    def test_large_sequence_length(self):
        # Test with sequence length at the maximum
        model = LayoutLMv3Backbone(**self.init_kwargs)

        seq_len = 128  # max_sequence_length
        input_data = {
            "token_ids": ops.cast(
                keras.random.uniform(
                    shape=(1, seq_len), minval=0, maxval=1000, dtype="float32"
                ),
                "int32",
            ),
            "padding_mask": keras.ops.ones((1, seq_len), dtype="int32"),
            "bbox": ops.cast(
                keras.random.uniform(
                    shape=(1, seq_len, 4),
                    minval=0,
                    maxval=1000,
                    dtype="float32",
                ),
                "int32",
            ),
        }

        output = model(input_data)
        expected_shape = [1, seq_len, 64]
        self.assertEqual(list(output.shape), expected_shape)
