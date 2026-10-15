from unittest.mock import patch

import keras
import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionAttention,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoderLayer,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionMLP,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionPatchEmbedder,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionRotaryEmbedding,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    _per_patch_coordinates,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    _window_layout,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class MuseGlimmerVisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_layers": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "intermediate_size": 16,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "layer_types": ["window_attention", "full_attention"],
        }
        self.encoder = MuseGlimmerVisionEncoder(**self.init_kwargs)

    def test_call_with_single_image(self):
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        total_patches = 1 * 4 * 4
        patch_dim = 2 * 3 * 2 * 2  # patch_temporal * 3 * patch_size**2
        pixel_values = np.random.randn(total_patches, patch_dim).astype(
            "float32"
        )
        output = self.encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        # 16 patches merged 2x2 -> 4 tokens, out_hidden_size = 8 * 2**2 = 32.
        self.assertEqual(ops.shape(output), (4, 32))

    def test_call_with_unbatched_grid(self):
        patch_dim = 2 * 3 * 2 * 2
        pixel_values = np.random.randn(16, patch_dim).astype("float32")
        output = self.encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor([1, 4, 4], dtype="int32"),
        )
        self.assertEqual(ops.shape(output), (4, 32))

    def test_call_with_multiple_images(self):
        grid_thw = np.array([[1, 4, 4], [1, 2, 2]], dtype="int32")
        total_patches = 1 * 4 * 4 + 1 * 2 * 2
        patch_dim = 2 * 3 * 2 * 2
        pixel_values = np.random.randn(total_patches, patch_dim).astype(
            "float32"
        )
        output = self.encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        # (16 + 4) patches merged 2x2 -> 4 + 1 = 5 tokens.
        self.assertEqual(ops.shape(output), (5, 32))

    def test_mlp_basics(self):
        self.run_layer_test(
            cls=MuseGlimmerVisionMLP,
            init_kwargs={"hidden_size": 8, "intermediate_size": 16},
            input_data=ops.zeros((2, 5, 8)),
            expected_output_shape=(2, 5, 8),
            expected_num_trainable_weights=4,
        )

    def test_patch_embedder_serialization(self):
        layer = MuseGlimmerVisionPatchEmbedder(
            hidden_size=8, pos_emb_height=4, pos_emb_width=4
        )
        self.run_serialization_test(layer)

    def test_rotary_embedding_serialization(self):
        layer = MuseGlimmerVisionRotaryEmbedding(head_dim=8, theta=500.0)
        self.run_serialization_test(layer)

    def test_attention_serialization(self):
        layer = MuseGlimmerVisionAttention(hidden_size=8, num_heads=2)
        self.run_serialization_test(layer)

    def test_encoder_layer_serialization(self):
        layer = MuseGlimmerVisionEncoderLayer(
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            layer_norm_eps=1e-6,
        )
        self.run_serialization_test(layer)

    def test_position_embedding_interpolation_uses_half_pixel_coordinates(
        self,
    ):
        patch_embedder = MuseGlimmerVisionPatchEmbedder(
            hidden_size=1,
            pos_emb_height=4,
            pos_emb_width=4,
        )
        patch_embedder.build((None, 1))
        patch_embedder.position_embedding_table.embeddings.assign(
            np.arange(16, dtype="float32")[:, np.newaxis]
        )

        output = ops.convert_to_numpy(
            patch_embedder._bilinear_position_embeddings(
                _per_patch_coordinates(
                    np.array([[1, 5, 5]], dtype="int32"), num_patches=25
                )
            )
        )[:, 0]

        expected = np.array(
            [
                0.0,
                0.63,
                1.35,
                2.07,
                2.43,
                2.52,
                3.50,
                4.30,
                5.10,
                5.22,
                5.40,
                6.70,
                7.50,
                8.30,
                8.10,
                8.28,
                9.90,
                10.70,
                11.50,
                10.98,
                9.72,
                11.43,
                12.15,
                12.87,
                12.15,
            ],
            dtype="float32",
        )

        self.assertAllClose(output, expected, atol=1e-6, rtol=1e-6)

    def test_rotary_positions_use_one_based_coordinates(self):
        cos, sin = self.encoder._rot_pos_emb(
            _per_patch_coordinates(
                np.array([[1, 2, 2]], dtype="int32"), num_patches=4
            )
        )
        expected_positions = np.array(
            [[1, 1], [2, 1], [1, 2], [2, 2]], dtype="float32"
        )
        expected_cos = np.cos(expected_positions)
        expected_sin = np.sin(expected_positions)
        expected_cos = np.stack(
            [
                expected_cos[:, 0],
                expected_cos[:, 1],
                expected_cos[:, 0],
                expected_cos[:, 1],
            ],
            axis=-1,
        )
        expected_sin = np.stack(
            [
                expected_sin[:, 0],
                expected_sin[:, 1],
                expected_sin[:, 0],
                expected_sin[:, 1],
            ],
            axis=-1,
        )
        self.assertAllClose(cos, expected_cos)
        self.assertAllClose(sin, expected_sin)

    def test_window_layout_pads_ragged_windows(self):
        grid_thw = ops.convert_to_tensor(np.array([[1, 3, 5]], dtype="int32"))
        window_index, reverse_indices, window_segment_id = _window_layout(
            grid_thw,
            _per_patch_coordinates(grid_thw, num_patches=15),
            window_patches=4,
        )

        expected_window_index = np.array(
            [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 4, 9, 14],
            dtype="int32",
        )
        self.assertAllEqual(window_index, expected_window_index)
        self.assertAllEqual(reverse_indices, np.argsort(expected_window_index))
        self.assertAllEqual(window_segment_id, np.repeat([0, 1], [12, 3]))

    def test_window_layout_offsets_each_video_frame(self):
        grid_thw = ops.convert_to_tensor(np.array([[2, 3, 5]], dtype="int32"))
        window_index, reverse_indices, window_segment_id = _window_layout(
            grid_thw,
            _per_patch_coordinates(grid_thw, num_patches=30),
            window_patches=4,
        )

        frame_index = np.array(
            [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 4, 9, 14],
            dtype="int32",
        )
        expected_window_index = np.concatenate([frame_index, frame_index + 15])
        self.assertAllEqual(window_index, expected_window_index)
        self.assertAllEqual(reverse_indices, np.argsort(expected_window_index))
        self.assertAllEqual(
            window_segment_id, np.repeat([0, 1, 2, 3], [12, 3, 12, 3])
        )

    def _capped_encoder(self, **caps):
        encoder = MuseGlimmerVisionEncoder(**self.init_kwargs, **caps)
        encoder.build()
        self.encoder.build()
        encoder.set_weights(self.encoder.get_weights())
        return encoder

    def _traced(self, encoder):
        """Return a function that runs `encoder` in a real trace."""
        patch_dim = 2 * 3 * 2 * 2
        backend = keras.config.backend()
        if backend == "jax":
            import jax

            return jax.jit(lambda p, g: encoder(p, g))
        if backend == "tensorflow":
            return tf.function(
                lambda p, g: encoder(p, g),
                input_signature=[
                    tf.TensorSpec([None, patch_dim], tf.float32),
                    tf.TensorSpec([None, 3], tf.int32),
                ],
            )
        self.skipTest("Only JAX and TensorFlow trace the encoder.")

    def test_traced_paths_match_eager_attention(self):
        # Eager calls batch exact-length segments. Traced calls use the
        # padded buffers when the caps are set, or one masked attention
        # when the caps are `None`. All three paths must match.
        capped = self._capped_encoder(
            max_num_windows=16, max_num_frames=4, max_frame_size=64
        )
        traced_masked = self._traced(self.encoder)
        traced_capped = self._traced(capped)
        patch_dim = 2 * 3 * 2 * 2
        # An image with ragged windows, and a two-frame video.
        for grid in ([[1, 6, 10]], [[2, 4, 6]], [[1, 6, 10], [2, 4, 6]]):
            grid_thw = np.array(grid, dtype="int32")
            num_patches = int(np.prod(grid_thw, axis=-1).sum())
            pixel_values = np.random.randn(num_patches, patch_dim).astype(
                "float32"
            )
            eager_output = self.encoder(pixel_values, grid_thw)
            masked_output = traced_masked(pixel_values, grid_thw)
            capped_output = traced_capped(pixel_values, grid_thw)
            self.assertAllClose(
                masked_output, eager_output, atol=1e-5, rtol=1e-5
            )
            self.assertAllClose(
                capped_output, eager_output, atol=1e-5, rtol=1e-5
            )

    def test_eager_call_groups_segments_by_length(self):
        patch_dim = 2 * 3 * 2 * 2
        grid_thw = np.array([[1, 6, 10], [2, 4, 6]], dtype="int32")
        pixel_values = np.random.randn(108, patch_dim).astype("float32")
        window_patches = max(
            self.encoder.window_size // self.encoder.patch_size, 1
        )
        window_groups, frame_groups = self.encoder._eager_segment_groups(
            grid_thw, window_patches
        )
        self.assertEqual(frame_groups[1], ((2, 24), (1, 60)))
        num_windows = sum(count for count, _ in window_groups[1])
        expected_num_windows = sum(
            t * -(-h // window_patches) * -(-w // window_patches)
            for t, h, w in grid_thw.tolist()
        )
        self.assertEqual(num_windows, expected_num_windows)
        # Count the calls of the traced-path methods. They wrap the real
        # methods, so the real code path still runs.
        num_calls = {"masked": 0, "padded": 0}
        masked_attention = MuseGlimmerVisionAttention._masked_full_attention
        padded_attention = MuseGlimmerVisionAttention._padded_segment_attention

        def counting_masked(attention, *args, **kwargs):
            num_calls["masked"] += 1
            return masked_attention(attention, *args, **kwargs)

        def counting_padded(attention, *args, **kwargs):
            num_calls["padded"] += 1
            return padded_attention(attention, *args, **kwargs)

        with (
            patch.object(
                MuseGlimmerVisionAttention,
                "_masked_full_attention",
                counting_masked,
            ),
            patch.object(
                MuseGlimmerVisionAttention,
                "_padded_segment_attention",
                counting_padded,
            ),
        ):
            self.encoder(pixel_values, grid_thw)
        self.assertEqual(num_calls, {"masked": 0, "padded": 0})

    def test_padding_caps_too_small_raise(self):
        patch_dim = 2 * 3 * 2 * 2
        grid_thw = np.array([[2, 4, 6]], dtype="int32")
        pixel_values = np.random.randn(48, patch_dim).astype("float32")
        for caps in (
            {"max_num_windows": 3},
            {"max_num_frames": 1, "max_frame_size": 24},
            {"max_num_frames": 2, "max_frame_size": 16},
        ):
            with self.assertRaises(ValueError):
                self._capped_encoder(**caps)(pixel_values, grid_thw)

    def test_frame_caps_must_be_set_together(self):
        with self.assertRaises(ValueError):
            MuseGlimmerVisionEncoder(**self.init_kwargs, max_num_frames=2)

    def test_block_norms_use_fixed_epsilon(self):
        # Hugging Face applies `layer_norm_eps` to `ln_pre` and `ln_post`
        # only. The block norms use a fixed 1e-5.
        encoder = MuseGlimmerVisionEncoder(
            **dict(self.init_kwargs, layer_norm_eps=1e-2)
        )
        self.assertEqual(encoder.ln_pre.epsilon, 1e-2)
        self.assertEqual(encoder.ln_post.epsilon, 1e-2)
        for block in encoder.blocks:
            self.assertEqual(block.layer_norm_eps, 1e-5)

    def test_invalid_layer_types_raise(self):
        for layer_types in (
            ["window_attention"],
            ["window_attention"] * 3,
            ["window_attention", "sliding_attention"],
        ):
            with self.assertRaisesRegex(ValueError, "layer_types"):
                MuseGlimmerVisionEncoder(
                    **dict(self.init_kwargs, layer_types=layer_types)
                )

    def test_serialization(self):
        self.run_serialization_test(self.encoder)
