import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_decoder import (
    MuseGlimmerTextDecoder,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerTextDecoderTest(TestCase):
    def test_forward_full_attention_nope(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_rope=False,
            sliding_window_size=None,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))

    def test_forward_sliding_window_rope(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_rope=True,
            sliding_window_size=2,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))

    def test_get_config(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        config = layer.get_config()
        restored = MuseGlimmerTextDecoder.from_config(config)
        self.assertEqual(restored.intermediate_dim, 16)
