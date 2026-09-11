from keras import ops

from keras_hub.src.models.phi3.phi3_rotary_embedding import (
    Phi3SuScaledRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class Phi3SuScaledRotaryEmbeddingTest(TestCase):
    def setUp(self):
        self.head_dim = 8
        self.pretraining_sequence_length = 8
        self.init_kwargs = {
            # A factor of 1.0 leaves the inverse frequencies untouched while
            # 4.0 scales them, so the two branches are distinguishable.
            "inverese_freq_short_factor": [1.0] * (self.head_dim // 2),
            "inverese_freq_long_factor": [4.0] * (self.head_dim // 2),
            "max_sequence_length": 64,
            "pretraining_sequence_length": self.pretraining_sequence_length,
        }

    def _rotate(self, sequence_length, start_index):
        layer = Phi3SuScaledRotaryEmbedding(**self.init_kwargs)
        inputs = ops.ones((1, sequence_length, 2, self.head_dim))
        return layer(inputs, start_index=start_index)

    def test_forward_pass_switches_on_sequence_length(self):
        length = self.pretraining_sequence_length
        short = self._rotate(length, start_index=0)
        long = self._rotate(length + 1, start_index=0)
        # Position 0 is unrotated under either factor, so compare position 1.
        self.assertNotAllClose(short[:, 1:2], long[:, 1:2])

    def test_cached_decoding_uses_long_factor_past_pretraining_length(self):
        # Same position on both sides, so the factor is the only variable.
        limit = self.pretraining_sequence_length
        full = self._rotate(limit + 1, start_index=0)
        cached = self._rotate(1, start_index=limit)
        self.assertAllClose(cached, full[:, limit : limit + 1])

    def test_cached_decoding_matches_uncached_forward_pass(self):
        # Below the crossover, cached and uncached must agree per position.
        limit = self.pretraining_sequence_length
        full = self._rotate(limit, start_index=0)
        for index in range(limit):
            self.assertAllClose(
                self._rotate(1, start_index=index),
                full[:, index : index + 1],
            )
