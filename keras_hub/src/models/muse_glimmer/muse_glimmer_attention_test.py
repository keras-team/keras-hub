import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_attention import (
    MuseGlimmerTextAttention,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerTextAttentionTest(TestCase):
    def test_forward_shape(self):
        init_kwargs = dict(
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            qk_scale_factor=3.87,
            use_rope=True,
            sliding_window_size=None,
        )
        self.run_serialization_test(MuseGlimmerTextAttention(**init_kwargs))
        layer = MuseGlimmerTextAttention(**init_kwargs)
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        mask = ops.ones((2, 5, 5), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))
        self.assertGreater(len(layer.trainable_weights), 0)

    def test_nope_layer_ignores_position(self):
        init_kwargs = dict(
            num_query_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            use_rope=False,
        )
        self.run_serialization_test(MuseGlimmerTextAttention(**init_kwargs))
        layer = MuseGlimmerTextAttention(**init_kwargs)
        x = ops.convert_to_tensor(np.random.randn(1, 4, 8).astype("float32"))
        mask = ops.ones((1, 4, 4), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (1, 4, 8))
        self.assertFalse(hasattr(layer, "rotary_embedding_layer"))

    def test_sliding_window(self):
        init_kwargs = dict(
            num_query_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            use_rope=True,
            sliding_window_size=2,
        )
        self.run_serialization_test(MuseGlimmerTextAttention(**init_kwargs))
        layer = MuseGlimmerTextAttention(**init_kwargs)
        x = ops.convert_to_tensor(np.random.randn(1, 6, 8).astype("float32"))
        mask = ops.ones((1, 6, 6), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (1, 6, 8))
