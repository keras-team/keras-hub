import numpy as np
import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 256,
        }
        self.input_data = {
            "token_ids": np.array([[1, 2, 3, 4, 5]]),
            "padding_mask": np.array([[1, 1, 1, 1, 1]]),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen2VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 5, 128),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLBackbone.presets:
            self.run_preset_test(
                cls=Qwen2VLBackbone,
                preset=preset,
                input_data=self.input_data,
            )


class Qwen2VLMultimodalBackboneTest(TestCase):
    """Tests for the backbone with a vision encoder attached."""

    def setUp(self):
        self.vision_encoder = Qwen2VLVisionEncoder(
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=64,
            out_dim=128,
            depth=2,
            num_heads=4,
            mlp_ratio=4,
            spatial_merge_size=2,
        )
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 256,
            # head_dim = 128/8 = 16 → sections must sum to 8.
            "mrope_section": [2, 3, 3],
            "vision_encoder": self.vision_encoder,
        }
        self.input_data = {
            "token_ids": np.array([[1, 2, 3, 4, 5]]),
            "padding_mask": np.array([[1, 1, 1, 1, 1]]),
        }

    def _vision_inputs(self, seq_len=5):
        """2 merged vision tokens from grid [2, 2, 2] at positions 1, 2."""
        grid_thw = np.array([[2, 2, 2]], dtype="int32")
        pixel_values = np.random.rand(8, 2, 14, 14, 3).astype("float32")
        vision_indices = np.array([1, 2], dtype="int32")
        # Sequential positions (equivalent to 1-D RoPE).
        position_ids = np.broadcast_to(
            np.arange(seq_len)[None, None, :], (1, 3, seq_len)
        ).astype("int32")
        return {
            "token_ids": np.array([[1, 999, 999, 4, 5]]),
            "padding_mask": np.array([[1, 1, 1, 1, 1]]),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": vision_indices,
            "position_ids": position_ids,
        }

    def test_multimodal_backbone_builds(self):
        model = Qwen2VLBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_forward(self):
        backbone = Qwen2VLBackbone(**self.init_kwargs)
        output = backbone(self._vision_inputs())
        self.assertEqual(tuple(output.shape), (1, 5, 128))

    def test_backbone_text_only_injection(self):
        """Multimodal backbone accepts text-only inputs via __call__."""
        backbone = Qwen2VLBackbone(**self.init_kwargs)
        output = backbone(self.input_data)
        self.assertEqual(tuple(output.shape), (1, 5, 128))

    def test_backbone_mrope_positions(self):
        """M-RoPE position ids alter outputs vs. plain sequential RoPE."""
        backbone = Qwen2VLBackbone(**self.init_kwargs)
        inputs = self._vision_inputs()
        out_flat = backbone(inputs)

        # The two vision tokens occupy frames t=1 and t=2 (grid t=2),
        # with h=w=1; text continues from 1 + max(1, 1) = 2.
        pos_mrope = np.array(
            [
                [
                    [0, 1, 2, 2, 3],
                    [0, 1, 1, 2, 3],
                    [0, 1, 1, 2, 3],
                ]
            ],
            dtype="int32",
        )
        out_mrope = backbone({**inputs, "position_ids": pos_mrope})
        self.assertNotAllClose(out_flat, out_mrope)
