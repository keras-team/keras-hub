import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
        }
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = Qwen3_5Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        # Pinned to exactly 2 devices (not len(devices)): the default test
        # config's num_key_value_heads=2 is intentionally left as-is (not
        # divisible by every host's device count) to regression-test that
        # key/value kernels are left replicated rather than sharded -- see
        # get_layout_map's comment.
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = Qwen3_5Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = Qwen3_5Backbone(**self.init_kwargs)

        for w in model.weights:
            if "token_embedding/embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "token_embedding/reverse_embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "self_attention/query/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "self_attention/key/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "self_attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "self_attention/attention_output/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, "batch")
                )
            if "feedforward_intermediate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "feedforward_output_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )


class Qwen3_5MultimodalBackboneTest(TestCase):
    """Tests for the backbone with vision encoder attached."""

    def setUp(self):
        self.vision_encoder = Qwen3_5VisionEncoder(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            in_channels=3,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=64,
        )
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "vision_encoder": self.vision_encoder,
            "mrope_section": [1, 1, 1],
        }

    def test_multimodal_backbone_builds(self):
        """Verify multimodal backbone constructs and has expected attributes."""
        model = Qwen3_5Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_forward(self):
        """Multimodal backbone forward pass with vision inputs."""
        model = Qwen3_5Backbone(**self.init_kwargs)

        # Build vision inputs: 1 image, 4x4 grid, patch_size=4,
        # temporal_patch_size=2. After spatial merge (2x2), 4 tokens.
        grid_thw = np.array([[[1, 4, 4]]], dtype="int32")  # (1, 1, 3)
        total_patches = 1 * 4 * 4  # 16
        # Batched pixel_values: (1, total_patches, T, pH, pW, C)
        pixel_values = np.random.randn(1, total_patches, 2, 4, 4, 3).astype(
            "float32"
        )

        seq_len = 10
        # Place 4 vision tokens at positions 2,3,4,5.
        vision_indices = np.array([[2, 3, 4, 5]], dtype="int32")

        input_data = {
            "token_ids": np.ones((1, seq_len), dtype="int32"),
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": vision_indices,
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (1, seq_len, 16))

    def test_vision_encoder_standalone(self):
        """Test vision encoder produces correct output shape standalone."""
        encoder = self.vision_encoder

        # 16 patches, spatial_merge_size=2 → 4 merged tokens.
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        total_patches = 1 * 4 * 4
        pixel_values = np.random.randn(total_patches, 2, 4, 4, 3).astype(
            "float32"
        )

        output = encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        self.assertEqual(ops.shape(output), (4, 16))

    def test_interleave_embeddings(self):
        """Test that interleave layer correctly scatters vision tokens."""
        model = Qwen3_5Backbone(**self.init_kwargs)

        batch_size = 1
        seq_len = 8
        hidden_dim = 16

        text_emb = ops.zeros((batch_size, seq_len, hidden_dim))
        vision_emb = np.ones((2, hidden_dim), dtype="float32")
        indices = ops.convert_to_tensor([1, 3], dtype="int32")

        result = model.interleave_embeddings(
            image_embeddings=vision_emb,
            text_embeddings=text_emb,
            vision_indices=indices,
        )
        self.assertEqual(ops.shape(result), (batch_size, seq_len, hidden_dim))

        result_np = ops.convert_to_numpy(result)
        np.testing.assert_allclose(result_np[0, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 1, :], 1.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 2, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 3, :], 1.0, atol=1e-6)
