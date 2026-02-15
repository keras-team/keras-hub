"""Tests for Qwen2-VL Backbone."""

import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLBackboneTextOnlyTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.seq_length = 16
        self.hidden_dim = 32
        self.head_dim = self.hidden_dim // 4  # 8

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": 64,
            "mrope_section": [1, 1, 2],  # sums to head_dim // 2 = 4
        }

        # For M-RoPE, position_ids shape is (batch, seq_len, 3)
        # For text-only, all 3 components are the same sequential IDs
        pos_ids = np.broadcast_to(
            np.arange(self.seq_length)[None, :, None],
            (self.batch_size, self.seq_length, 3),
        ).astype("int32")

        self.input_data = {
            "token_ids": np.random.randint(
                0,
                self.vocabulary_size,
                (self.batch_size, self.seq_length),
            ).astype("int32"),
            "padding_mask": np.ones(
                (self.batch_size, self.seq_length),
                dtype="int32",
            ),
            "mrope_position_ids": pos_ids,
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen2VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.seq_length,
                self.hidden_dim,
            ),
        )

    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = Qwen2VLBackbone(**self.init_kwargs)
        # Check that model has expected number of transformer layers
        self.assertEqual(len(model.transformer_layers), 2)
        # Check layer norm exists
        self.assertIsNotNone(model.layer_norm)


class Qwen2VLBackboneMultimodalTest(TestCase):
    def test_multimodal_forward(self):
        hidden_dim = 32
        batch_size = 2
        seq_length = 24

        vision_encoder = Qwen2VLVisionEncoder(
            hidden_size=hidden_dim,
            embed_dim=16,
            depth=1,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            mlp_ratio=2.0,
            spatial_merge_size=2,
        )

        model = Qwen2VLBackbone(
            vocabulary_size=256,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=hidden_dim,
            intermediate_dim=64,
            mrope_section=[1, 1, 2],
            vision_encoder=vision_encoder,
        )

        # Per sample: 16 patches -> 4 merged vision tokens.
        images = np.random.rand(batch_size, 16, 3, 2, 2, 2).astype("float32")
        grid_thw = np.array(
            [
                [[1, 4, 4]],
                [[1, 4, 4]],
            ],
            dtype="int32",
        )
        vision_indices = np.array(
            [
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype="int32",
        )
        mrope_position_ids = np.broadcast_to(
            np.arange(seq_length)[None, :, None],
            (batch_size, seq_length, 3),
        ).astype("int32")

        inputs = {
            "token_ids": np.random.randint(
                0, 256, (batch_size, seq_length), dtype="int32"
            ),
            "padding_mask": np.ones((batch_size, seq_length), dtype="int32"),
            "mrope_position_ids": mrope_position_ids,
            "images": images,
            "vision_indices": vision_indices,
            "grid_thw": grid_thw,
        }

        output = model(inputs)
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_dim))
