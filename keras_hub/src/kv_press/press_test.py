import keras
import numpy as np
from keras import ops

from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.press import KVCachePress
from keras_hub.src.kv_press.random_press import RandomPress
from keras_hub.src.kv_press.streaming_llm_press import StreamingLLMPress
from keras_hub.src.tests.test_case import TestCase


def make_cache(batch_size, num_layers, seq_len, num_heads, head_dim, key_fn):
    """Build a `(batch, num_layers, 2, seq_len, num_heads, head_dim)` cache.

    `key_fn(position)` returns the head_dim-length key vector to use at a
    given sequence position (shared across batch/layer/head), so tests can
    control importance scores (e.g. via key norm) deterministically. Values
    are left as zeros since none of the presses under test score on values.
    """
    keys = np.zeros((batch_size, num_layers, seq_len, num_heads, head_dim))
    for pos in range(seq_len):
        keys[:, :, pos, :, :] = key_fn(pos)
    values = np.zeros_like(keys)
    cache = np.stack([keys, values], axis=2)
    return ops.convert_to_tensor(cache, dtype="float32")


class KVCachePressTest(TestCase):
    def test_invalid_compression_ratio(self):
        with self.assertRaises(ValueError):
            KnormPress(compression_ratio=1.0)
        with self.assertRaises(ValueError):
            KnormPress(compression_ratio=-0.1)

    def test_zero_compression_ratio_is_a_no_op(self):
        cache = make_cache(2, 2, 6, 2, 3, lambda pos: pos)
        press = KnormPress(compression_ratio=0.0)
        compressed = press.compress(cache)
        self.assertAllClose(compressed, cache)

    def test_output_shape(self):
        cache = make_cache(2, 3, 8, 2, 4, lambda pos: pos)
        for press in (
            RandomPress(compression_ratio=0.5),
            KnormPress(compression_ratio=0.5),
            StreamingLLMPress(compression_ratio=0.5, n_sink=1),
        ):
            compressed = press.compress(cache)
            self.assertEqual(tuple(compressed.shape), (2, 3, 2, 4, 2, 4))

    def test_padding_positions_are_evicted_first(self):
        if keras.config.backend() != "torch":
            self.skipTest("Padding-aware compression is torch-only.")
        seq_len = 10

        # A press that (absent padding-awareness) would keep the *last*
        # `keep_len` positions -- i.e. exactly the padding for a short row.
        class LastPositionsPress(KVCachePress):
            def score(self, keys, values, keep_len, padding_mask=None):
                positions = ops.arange(seq_len, dtype="float32")
                shape = ops.shape(keys)
                score = ops.reshape(positions, (1, 1, 1, seq_len))
                return ops.broadcast_to(
                    score, (shape[0], shape[1], shape[3], seq_len)
                )

        cache = make_cache(2, 1, seq_len, 1, 1, lambda pos: [pos])
        # Both rows are real through position 5; row 1 is padding after it.
        padding_mask = ops.convert_to_tensor(
            [[True] * 10, [True] * 6 + [False] * 4]
        )
        # prompt_len = min row length = 6, so keep_len = round(6 * 0.5) = 3
        # and 10 - 6 = 4 reserve slots are appended for generation.
        press = LastPositionsPress(compression_ratio=0.5)
        compressed = press.compress(cache, padding_mask=padding_mask)
        self.assertEqual(compressed.shape[3], 3 + 4)
        for row in (0, 1):
            kept = ops.convert_to_numpy(compressed[row, 0, 0, :3, 0, 0]).astype(
                int
            )
            # Positions 6-9 are padding for row 1 and outside the shared
            # prompt region for row 0, so neither row may retain them even
            # though the raw score favors them. Position 5 is force-kept as
            # the last prompt position; 3 and 4 are the next highest.
            self.assertAllEqual(kept, [3, 4, 5])
            reserve = ops.convert_to_numpy(compressed[row, 0, 0, 3:, 0, 0])
            self.assertAllClose(reserve, [0.0] * 4)

    def test_keep_len_is_a_fraction_of_the_real_prompt(self):
        if keras.config.backend() != "torch":
            self.skipTest("Padding-aware compression is torch-only.")
        cache = make_cache(1, 1, 10, 1, 1, lambda pos: pos)
        padding_mask = ops.convert_to_tensor([[True] * 8 + [False] * 2])
        press = KnormPress(compression_ratio=0.5)
        compressed = press.compress(cache, padding_mask=padding_mask)
        # prompt_len=8 -> keep_len = round(8 * 0.5) = 4, plus a 10 - 8 = 2
        # slot reserve for the tokens generation will write.
        self.assertEqual(compressed.shape[3], 4 + 2)

    def test_reserve_leaves_room_for_every_decode_step(self):
        # The compressed buffer must be able to hold the whole generation:
        # `keep_len` retained prompt slots plus one slot per decode step.
        # Without the reserve the decode loop wraps back over the retained
        # prompt and silently destroys it.
        if keras.config.backend() != "torch":
            self.skipTest("Padding-aware compression is torch-only.")
        seq_len, prompt_len = 10, 6
        cache = make_cache(1, 1, seq_len, 1, 1, lambda pos: pos)
        padding_mask = ops.convert_to_tensor(
            [[True] * prompt_len + [False] * (seq_len - prompt_len)]
        )
        for ratio in (0.25, 0.5, 0.75):
            compressed = KnormPress(compression_ratio=ratio).compress(
                cache, padding_mask=padding_mask
            )
            buffer_len = compressed.shape[3]
            # This is what `CausalLM._compress_layer_cache` callers derive.
            offset = seq_len - buffer_len
            keep_len = buffer_len - (seq_len - prompt_len)
            # First decode step lands on the last retained slot...
            self.assertEqual((prompt_len - 1) - offset, keep_len - 1)
            # ...and the last one still fits inside the buffer.
            self.assertLess((seq_len - 1) - offset, buffer_len)

    def test_compress_rejects_padding_mask_off_torch(self):
        if keras.config.backend() == "torch":
            self.skipTest("torch supports padding-aware compression.")
        cache = make_cache(1, 1, 10, 1, 1, lambda pos: pos)
        padding_mask = ops.convert_to_tensor([[True] * 8 + [False] * 2])
        with self.assertRaises(NotImplementedError):
            KnormPress(compression_ratio=0.5).compress(
                cache, padding_mask=padding_mask
            )
