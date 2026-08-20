from keras import ops

from keras_hub.src.models.phi3.phi3_rotary_embedding import (
    Phi3SuScaledRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class Phi3SuScaledRotaryEmbeddingTest(TestCase):
    def _build_layer(self, pretraining_sequence_length=4):
        return Phi3SuScaledRotaryEmbedding(
            inverese_freq_short_factor=[1.0, 1.0],
            inverese_freq_long_factor=[8.0, 8.0],
            max_sequence_length=64,
            pretraining_sequence_length=pretraining_sequence_length,
        )

    def test_decode_step_matches_a_prompt_of_the_same_length(self):
        """Which factor applies is a property of how far the sequence has
        run, so decoding the token at position k must match position k of a
        prompt holding k + 1 tokens. Keying on the input's own length gives
        a single decoded token short-context frequencies forever."""
        layer = self._build_layer(pretraining_sequence_length=4)
        token = ops.ones((1, 1, 2, 4))

        for index in range(8):
            prompt = ops.ones((1, index + 1, 2, 4))
            self.assertAllClose(
                layer(prompt)[:, index : index + 1],
                layer(token, start_index=index),
            )

    def test_long_factor_takes_over_past_the_pretraining_length(self):
        layer = self._build_layer(pretraining_sequence_length=4)
        token = ops.ones((1, 1, 2, 4))

        within = layer(token, start_index=2)
        beyond = layer(token, start_index=100)

        self.assertNotAllClose(within, beyond)

    def test_explicit_positions_match_start_index(self):
        layer = self._build_layer()
        inputs = ops.ones((1, 1, 2, 4))

        stepped = layer(inputs, start_index=3)
        unbatched = layer(inputs, positions=ops.convert_to_tensor([3.0]))
        batched = layer(inputs, positions=ops.convert_to_tensor([[3.0]]))

        self.assertAllClose(stepped, unbatched)
        self.assertAllClose(stepped, batched)

    def test_positions_are_per_token(self):
        """Two tokens given unrelated positions must be rotated by their
        own angles, not by a shared range."""
        layer = self._build_layer()
        inputs = ops.ones((1, 2, 2, 4))

        scattered = layer(inputs, positions=ops.convert_to_tensor([[5.0, 9.0]]))
        at_five = layer(ops.ones((1, 1, 2, 4)), start_index=5)

        self.assertAllClose(scattered[:, 0:1], at_five)
