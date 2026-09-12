import numpy as np
from keras import ops
from keras import random

from keras_hub.src.layers.modeling.multimodal_rotary_embedding import (
    MultimodalRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class MultimodalRotaryEmbeddingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=MultimodalRotaryEmbedding,
            init_kwargs={
                "mrope_section": (2, 2, 2),
                "max_wavelength": 10000,
                "scaling_factor": 1.0,
                "attention_scaling": 1.0,
            },
            input_data=random.uniform(shape=(2, 4, 12)),
            expected_output_shape=(2, 4, 12),
        )

    def test_zero_positions_identity(self):
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 12
        layer = MultimodalRotaryEmbedding(mrope_section=(2, 2, 2))
        query = random.uniform(shape=(batch, seq_len, num_heads, head_dim))
        key = random.uniform(shape=(batch, seq_len, num_heads, head_dim))

        position_ids = np.zeros((3, batch, seq_len), dtype=np.int32)
        q_out, k_out = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids
        )

        self.assertAllClose(q_out, query, atol=1e-5)
        self.assertAllClose(k_out, key, atol=1e-5)

    def test_output_correct_values(self):
        # Reference values computed from the HF Qwen3-Omni-MoE implementation:
        # from transformers.models.qwen3_omni_moe import (
        #     modeling_qwen3_omni_moe as hf_model
        # )
        # hf_model.Qwen3OmniMoeThinkerTextRotaryEmbedding, apply_rotary_pos_emb
        # mrope_section=(2,2,2), rope_theta=10000, head_dim=12
        # query = key = ones((1, 2, 1, 12))
        # position_ids: T=[0,1], H=[0,0], W=[0,1]  shape (3, 1, 2)
        layer = MultimodalRotaryEmbedding(mrope_section=(2, 2, 2))
        query = ops.ones((1, 2, 1, 12))
        key = ops.ones((1, 2, 1, 12))
        # fmt: off
        position_ids = np.array([[[0, 1]], [[0, 0]], [[0, 1]]], dtype=np.int32)
        # fmt: on
        q_out, k_out = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids
        )

        # pos=0: T=0, H=0, W=0 → all cos=1, sin=0, output = input
        self.assertAllClose(q_out[0, 0, 0, :], [1.0] * 12, atol=1e-5)
        # pos=1: T=1, H=0, W=1 — mixed sections produce non-trivial rotation
        # fmt: off
        expected_pos1 = [
            -0.3012, 1.0, 0.9525, 0.9900, 1.0, 0.9995,
             1.3818, 1.0, 1.0453, 1.0100, 1.0, 1.0005,
        ]
        # fmt: on
        self.assertAllClose(q_out[0, 1, 0, :], expected_pos1, atol=1e-3)
        # query == key == ones so both outputs must be equal
        self.assertAllClose(q_out, k_out)

    def test_attention_scaling(self):
        # Reference values from HF implementation, attention_scaling=2.0.
        # query = key = ones((1, 2, 1, 12))
        # position_ids: T=[0,1], H=[0,1], W=[0,1]  (all-same, text-only)
        layer = MultimodalRotaryEmbedding(
            mrope_section=(2, 2, 2), attention_scaling=2.0
        )
        query = ops.ones((1, 2, 1, 12))
        key = ops.ones((1, 2, 1, 12))
        position_ids = np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int32)

        q_out, _ = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids
        )

        # fmt: off
        expected_pos1 = [
            -0.6023, 1.5262, 1.9050, 1.9799, 1.9957, 1.9991,
             2.7635, 2.3813, 2.0906, 2.0199, 2.0043, 2.0009,
        ]
        # fmt: on
        self.assertAllClose(q_out[0, 1, 0, :], expected_pos1, atol=1e-3)

    def test_multimodal_differs_from_text_only(self):
        layer = MultimodalRotaryEmbedding(mrope_section=(2, 2, 2))
        query = random.uniform(shape=(1, 4, 2, 12))
        key = random.uniform(shape=(1, 4, 2, 12))

        text_pos = np.array([[[0, 1, 2, 3]]], dtype=np.int32)
        position_ids_text = np.concatenate(
            [text_pos, text_pos, text_pos], axis=0
        )
        # Image tokens: H and W vary independently from T.
        position_ids_image = np.array(
            [[[0, 1, 2, 3]], [[0, 0, 1, 1]], [[0, 1, 0, 1]]], dtype=np.int32
        )

        q_text, _ = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids_text
        )
        q_image, _ = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids_image
        )

        self.assertNotAllClose(
            ops.convert_to_numpy(q_text), ops.convert_to_numpy(q_image)
        )

    def test_invalid_mrope_section_raises(self):
        with self.assertRaises(ValueError):
            MultimodalRotaryEmbedding(mrope_section=(2, 2))

        with self.assertRaises(ValueError):
            MultimodalRotaryEmbedding(mrope_section=(2, 2, 2, 2))

    def test_key_none_returns_rotated_query_only(self):
        layer = MultimodalRotaryEmbedding(mrope_section=(2, 2, 2))
        query = random.uniform(shape=(2, 3, 4, 12))
        key = random.uniform(shape=(2, 3, 4, 12))
        position_ids = np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
            ],
            dtype=np.int32,
        )

        q_both, k_both = layer.apply_multimodal_rotary_embedding(
            query, key, position_ids
        )
        q_only, k_only = layer.apply_multimodal_rotary_embedding(
            query, None, position_ids
        )

        self.assertIsNone(k_only)
        self.assertIsNotNone(k_both)
        self.assertAllClose(q_both, q_only)
