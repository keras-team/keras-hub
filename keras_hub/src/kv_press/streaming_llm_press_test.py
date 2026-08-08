import keras
from keras import ops

from keras_hub.src.kv_press.press_test import make_cache
from keras_hub.src.kv_press.streaming_llm_press import StreamingLLMPress
from keras_hub.src.tests.test_case import TestCase


class StreamingLLMPressTest(TestCase):
    def test_streaming_llm_press_keeps_sink_and_recent(self):
        seq_len = 10
        # Encode the position itself into the key so we can check identity.
        cache = make_cache(1, 1, seq_len, 1, 1, lambda pos: [pos])
        # keep_len = round(10 * 0.5) = 5, n_sink = 2 -> n_recent = 3.
        # Expected kept positions, in order: 0, 1, 7, 8, 9.
        press = StreamingLLMPress(compression_ratio=0.5, n_sink=2)
        compressed = press.compress(cache)
        kept_positions = ops.convert_to_numpy(
            compressed[0, 0, 0, :, 0, 0]
        ).astype(int)
        self.assertAllEqual(kept_positions, [0, 1, 7, 8, 9])

    def test_recent_window_is_relative_to_real_prompt_length(self):
        # Only the first 8 of 10 positions are real; the rest is unwritten
        # padding. "Recent" must track the tail of the *real* 8 tokens, not
        # the tail of the 10-slot buffer (which is partly padding here) --
        # otherwise this press would silently degrade to "sink tokens plus
        # arbitrary tie-broken real tokens" for any prompt shorter than the
        # buffer, which is the common case.
        if keras.config.backend() != "torch":
            self.skipTest("Padding-aware compression is torch-only.")
        seq_len = 10
        cache = make_cache(1, 1, seq_len, 1, 1, lambda pos: [pos])
        padding_mask = ops.convert_to_tensor([[True] * 8 + [False] * 2])
        press = StreamingLLMPress(compression_ratio=0.5, n_sink=1)
        compressed = press.compress(cache, padding_mask=padding_mask)
        # prompt_len=8 -> keep_len = round(8 * 0.5) = 4, followed by a
        # 10 - 8 = 2 slot reserve for the tokens generation will write.
        self.assertEqual(compressed.shape[3], 4 + 2)
        kept = ops.convert_to_numpy(compressed[0, 0, 0, :4, 0, 0]).astype(int)
        # n_sink=1, n_recent=3 -> sink {0}, recent real tail {5, 6, 7}.
        self.assertAllEqual(kept, [0, 5, 6, 7])
