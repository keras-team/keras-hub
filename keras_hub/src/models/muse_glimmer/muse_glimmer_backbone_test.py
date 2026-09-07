import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 4,
            "sliding_window_size": 4,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        }
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    # TODO: add HF parity test once weights are converted (requires a real
    # `meta-models/Muse-Glimmer-30B` checkpoint and cannot be run here).


class MuseGlimmerMultimodalBackboneTest(TestCase):
    def setUp(self):
        self.vision_encoder = MuseGlimmerVisionEncoder(
            num_layers=2,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            patch_size=2,
            patch_temporal=2,
            merge_size=2,
            pos_emb_height=4,
            pos_emb_width=4,
            layer_types=["window_attention", "full_attention"],
        )
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 32,
            "head_dim": 8,
            "sliding_window_size": 4,
            "layer_types": ["sliding_attention", "full_attention"],
            "vision_encoder": self.vision_encoder,
            "projector_hidden_dim": 16,
        }

    def test_multimodal_backbone_builds(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_forward(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)

        grid_thw = np.array([[[1, 4, 4]]], dtype="int32")  # (1, 1, 3)
        total_patches = 1 * 4 * 4
        patch_dim = 2 * 3 * 2 * 2  # patch_temporal * 3 * patch_size**2
        pixel_values = np.random.randn(1, total_patches, patch_dim).astype(
            "float32"
        )

        seq_len = 10
        vision_indices = np.array([[2, 3, 4, 5]], dtype="int32")

        input_data = {
            "token_ids": np.ones((1, seq_len), dtype="int32"),
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": vision_indices,
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (1, seq_len, 32))

    def test_text_only_call_on_multimodal_backbone(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        output = model(
            {
                "token_ids": np.ones((1, 5), dtype="int32"),
                "padding_mask": np.ones((1, 5), dtype="int32"),
            }
        )
        self.assertEqual(ops.shape(output), (1, 5, 32))
