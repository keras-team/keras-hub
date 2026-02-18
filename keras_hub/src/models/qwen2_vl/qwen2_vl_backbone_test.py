import numpy as np
import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
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
            "vision_patch_size": 14,
            "vision_temporal_patch_size": 2,
            "vision_in_channels": 3,
            "vision_embed_dim": 64,
            "vision_depth": 2,
            "vision_num_heads": 4,
            "spatial_merge_size": 2,
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

    def test_backbone_with_vision_inputs(self):
        """Test that the backbone correctly handles vision inputs."""
        # image_token_id defaults to 151655, but vocab_size=1000,
        # so use a custom backbone with image_token_id=999.
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["image_token_id"] = 999
        backbone = Qwen2VLBackbone(**init_kwargs)

        # Token sequence with 2 image placeholder tokens at positions 1,2.
        token_ids = np.array([[1, 999, 999, 4, 5]])
        padding_mask = np.array([[1, 1, 1, 1, 1]])

        # Vision: grid_thw = [1, 2, 2] → 4 raw patches.
        # After merger (spatial_merge_size=2): 4/4 = 1 merged token.
        # But we have 2 placeholder tokens, so we need 2 merged tokens.
        # grid_thw = [1, 4, 4] → 16 raw patches → 16/4 = 4 merged.
        # Use [2, 2, 2] → 8 raw patches → 8/4 = 2 merged tokens.
        grid_thw = np.array([[2, 2, 2]], dtype="int32")
        patch_flat_dim = 3 * 2 * 14 * 14  # in_channels * temporal * patch²
        total_patches = 8
        patch_values = np.random.rand(total_patches, patch_flat_dim).astype(
            "float32"
        )

        inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "patch_values": patch_values,
            "image_grid_thw": grid_thw,
        }
        output = backbone(inputs)
        self.assertEqual(output.shape, (1, 5, 128))

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
