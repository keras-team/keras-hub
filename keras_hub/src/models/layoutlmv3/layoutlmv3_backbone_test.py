import keras

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
            "token_ids": keras.random.uniform(
                shape=(2, 10), minval=0, maxval=1000, dtype="int32"
            ),
            "padding_mask": keras.ops.ones((2, 10), dtype="int32"),
            "bbox": keras.random.uniform(
                shape=(2, 10, 4), minval=0, maxval=1000, dtype="int32"
            ),
        }

    def test_backbone_basics(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        self.assertEqual(model.vocabulary_size, 1000)
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.num_heads, 2)
        self.assertEqual(model.intermediate_dim, 128)
        self.assertEqual(model.max_sequence_length, 128)
        self.assertEqual(model.spatial_embedding_dim, 32)

    def test_backbone_output_shape(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        output = model(self.input_data)
        # Output should be (batch_size, sequence_length, hidden_dim)
        expected_shape = [2, 10, 64]
        self.assertEqual(list(output.shape), expected_shape)

    def test_backbone_predict(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        output = model.predict(self.input_data)
        # Output should be (batch_size, sequence_length, hidden_dim)
        expected_shape = [2, 10, 64]
        self.assertEqual(list(output.shape), expected_shape)

    def test_saved_model(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        model_output = model(self.input_data)
        path = self.get_temp_dir()
        model.save(path)
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, LayoutLMv3Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_data)
        self.assertAllClose(model_output, restored_output)

    def test_get_config_and_from_config(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        config = model.get_config()
        restored_model = LayoutLMv3Backbone.from_config(config)

        # Check config was preserved
        self.assertEqual(restored_model.vocabulary_size, 1000)
        self.assertEqual(restored_model.hidden_dim, 64)
        self.assertEqual(restored_model.num_layers, 2)

    def test_compute_output_shape(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        batch_size = 3
        sequence_length = 5

        input_shapes = {
            "token_ids": (batch_size, sequence_length),
            "padding_mask": (batch_size, sequence_length),
            "bbox": (batch_size, sequence_length, 4),
        }

        output_shape = model.compute_output_shape(input_shapes)
        expected_shape = (batch_size, sequence_length, 64)
        self.assertEqual(output_shape, expected_shape)

    def test_different_sequence_lengths(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)

        # Test with different sequence length
        input_data = {
            "token_ids": keras.random.uniform(
                shape=(1, 5), minval=0, maxval=1000, dtype="int32"
            ),
            "padding_mask": keras.ops.ones((1, 5), dtype="int32"),
            "bbox": keras.random.uniform(
                shape=(1, 5, 4), minval=0, maxval=1000, dtype="int32"
            ),
        }

        output = model(input_data)
        expected_shape = [1, 5, 64]
        self.assertEqual(list(output.shape), expected_shape)

    def test_all_kwargs_in_config(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        config = model.get_config()

        # Ensure all init arguments are in the config
        for key, value in self.init_kwargs.items():
            self.assertEqual(config[key], value)

    def test_mixed_precision(self):
        # Test with mixed precision
        init_kwargs = {**self.init_kwargs, "dtype": "mixed_float16"}
        model = LayoutLMv3Backbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(output.dtype, "float16")

    def test_token_embedding_matrix_property(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)
        embeddings = model.token_embedding_matrix
        expected_shape = [1000, 64]  # vocabulary_size, hidden_dim
        self.assertEqual(list(embeddings.shape), expected_shape)

    def test_spatial_embeddings_initialization(self):
        model = LayoutLMv3Backbone(**self.init_kwargs)

        # Check that spatial embeddings have correct shapes
        x_embeddings = model.x_position_embedding.embeddings
        y_embeddings = model.y_position_embedding.embeddings
        h_embeddings = model.h_position_embedding.embeddings
        w_embeddings = model.w_position_embedding.embeddings

        expected_shape = [1024, 32]  # max_bbox_value, spatial_embedding_dim
        self.assertEqual(list(x_embeddings.shape), expected_shape)
        self.assertEqual(list(y_embeddings.shape), expected_shape)
        self.assertEqual(list(h_embeddings.shape), expected_shape)
        self.assertEqual(list(w_embeddings.shape), expected_shape)

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
            "token_ids": keras.random.uniform(
                shape=(1, seq_len), minval=0, maxval=1000, dtype="int32"
            ),
            "padding_mask": keras.ops.ones((1, seq_len), dtype="int32"),
            "bbox": keras.random.uniform(
                shape=(1, seq_len, 4), minval=0, maxval=1000, dtype="int32"
            ),
        }

        output = model(input_data)
        expected_shape = [1, seq_len, 64]
        self.assertEqual(list(output.shape), expected_shape)
