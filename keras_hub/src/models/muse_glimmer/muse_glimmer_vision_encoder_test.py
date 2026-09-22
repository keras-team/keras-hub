import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionMLP,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionPatchEmbedder,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    _window_layout,
)
from keras_hub.src.tests.test_case import TestCase


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
                np.array([[1, 5, 5]], dtype="int32"), num_patches=25
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
            np.array([[1, 2, 2]], dtype="int32")
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
        window_index, reverse_indices, window_segment_id = _window_layout(
            ops.convert_to_tensor(np.array([[1, 3, 5]], dtype="int32")),
            num_patches=15,
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
        window_index, reverse_indices, window_segment_id = _window_layout(
            ops.convert_to_tensor(np.array([[2, 3, 5]], dtype="int32")),
            num_patches=30,
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

    def test_serialization(self):
        self.run_serialization_test(self.encoder)
