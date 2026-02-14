"""Tests for Qwen2-VL Backbone."""

import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import (
    Qwen2VLBackbone,
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
                0, self.vocabulary_size,
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
