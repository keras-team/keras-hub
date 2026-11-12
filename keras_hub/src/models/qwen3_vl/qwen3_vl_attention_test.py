from unittest import mock

import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention
from keras_hub.src.tests.test_case import TestCase

HEAD_DIM = 16
SECTION = [4, 2, 2]


def _attention(**kwargs):
    config = dict(
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        rope_max_wavelength=10000.0,
        mrope_section=SECTION,
    )
    config.update(kwargs)
    return Qwen3VLAttention(**config)


class Qwen3VLAttentionTest(TestCase):
    def setUp(self):
        self.x = np.random.RandomState(0).normal(size=(2, 6, 32))
        self.x = self.x.astype("float32")

    def test_output_shape(self):
        layer = _attention()
        self.assertEqual(tuple(layer(self.x).shape), (2, 6, 32))

    def test_layer_basics(self):
        self.run_layer_test(
            cls=Qwen3VLAttention,
            init_kwargs=dict(
                num_query_heads=4,
                num_key_value_heads=2,
                head_dim=HEAD_DIM,
                rope_max_wavelength=10000.0,
                mrope_section=SECTION,
                dropout=0.1,
                layer_norm_epsilon=1e-5,
            ),
            input_data=self.x,
            expected_output_shape=(2, 6, 32),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_variables=1,  # dropout seed
        )

    def test_grouped_einsum_matches_fused_attention(self):
        layer = _attention()
        mask = np.tril(np.ones((6, 6), "bool"))[None].repeat(2, axis=0)
        fused = layer(self.x, attention_mask=mask)
        with mock.patch(
            "keras_hub.src.models.qwen3_vl.qwen3_vl_attention."
            "fused_attention_op_available",
            return_value=False,
        ):
            grouped = layer(self.x, attention_mask=mask)
        self.assertAllClose(fused, grouped, atol=1e-5)

    def test_dropout_applies_to_attention_probabilities(self):
        layer = _attention(dropout=0.5)
        # Inference uses no dropout; training drops attention weights.
        self.assertAllClose(layer(self.x), layer(self.x, training=False))
        dropped = ops.convert_to_numpy(layer(self.x, training=True))
        base = ops.convert_to_numpy(layer(self.x))
        self.assertGreater(np.abs(dropped - base).max(), 1e-4)

    def test_config_keeps_mrope_section(self):
        config = _attention().get_config()
        self.assertEqual(config["mrope_section"], SECTION)

    def test_mrope_section_must_match_head_dim(self):
        with self.assertRaises(ValueError):
            _attention(mrope_section=[4, 2, 1])

    def test_text_positions_match_default(self):
        layer = _attention()
        pos = np.tile(np.arange(6, dtype="int32"), (2, 3, 1))
        self.assertAllClose(layer(self.x), layer(self.x, position_ids=pos))

    def test_mrope_uses_distinct_channels(self):
        layer = _attention()
        rng = np.random.RandomState(2)
        pos = rng.randint(0, 50, size=(2, 3, 6)).astype("int32")
        # Permuting only the H channel changes the output.
        pos_h = pos.copy()
        pos_h[:, 1] = pos_h[:, 1, ::-1]
        self.assertNotAllClose(
            layer(self.x, position_ids=pos), layer(self.x, position_ids=pos_h)
        )

    def test_cache_matches_no_cache(self):
        layer = _attention()
        seq = 6
        rng = np.random.RandomState(3)
        pos = np.sort(rng.randint(0, 40, size=(2, 3, seq)), axis=-1)
        pos = pos.astype("int32")
        mask = np.tril(np.ones((seq, seq), dtype="bool"))
        mask = np.tile(mask, (2, 1, 1))
        full = layer(self.x, attention_mask=mask, position_ids=pos)

        cache = np.zeros((2, 2, seq, 2, HEAD_DIM), dtype="float32")
        outputs = []
        for i in range(seq):
            step_mask = np.zeros((2, 1, seq), dtype="bool")
            step_mask[:, :, : i + 1] = True
            out, cache = layer(
                self.x[:, i : i + 1],
                attention_mask=step_mask,
                cache=cache,
                cache_update_index=i,
                position_ids=pos[:, :, i : i + 1],
            )
            outputs.append(ops.convert_to_numpy(out))
        self.assertAllClose(np.concatenate(outputs, axis=1), full, atol=1e-5)
