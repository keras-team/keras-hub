import numpy as np
from keras import ops

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_attention import GPTNeoXAttention
from keras_hub.src.tests.test_case import TestCase


class GPTNeoXAttentionTest(TestCase):
    def _build_layer(self, rotary_percentage=1.0):
        layer = GPTNeoXAttention(
            num_heads=2,
            hidden_dim=16,
            rotary_percentage=rotary_percentage,
        )
        layer.build((1, 4, 16))
        return layer

    def _decode_step_by_step(self, layer, inputs, length):
        """Runs `inputs` one token at a time through the cached path."""
        cache = ops.zeros((1, 2, length, 2, 8))
        outputs = []
        for index in range(length):
            mask = np.zeros((1, 1, length), dtype=bool)
            # The token being decoded sees the cache filled so far.
            mask[0, 0, : index + 1] = True
            token = ops.convert_to_tensor(
                np.array(inputs)[:, index : index + 1]
            )
            output, cache = layer(
                token,
                attention_mask=ops.convert_to_tensor(mask),
                cache=cache,
                cache_update_index=index,
            )
            outputs.append(output)
        return ops.concatenate(outputs, axis=1)

    def test_cached_decoding_matches_full_sequence(self):
        """Each decoded token carries its own position, so generating one
        token at a time must match a single pass over the whole sequence.
        Rotating the query from position 0 every step silently breaks
        this."""
        layer = self._build_layer()
        inputs = ops.convert_to_tensor(
            np.random.default_rng(0).normal(size=(1, 4, 16)).astype("float32")
        )
        causal_mask = np.tril(np.ones((4, 4), dtype=bool))[None]

        full, _ = layer(
            inputs, attention_mask=ops.convert_to_tensor(causal_mask)
        )
        stepped = self._decode_step_by_step(layer, inputs, length=4)

        self.assertAllClose(full, stepped)

    def test_cached_decoding_matches_with_partial_rotary(self):
        """The same holds when only part of each head is rotated, which is
        GPT-NeoX's default."""
        layer = self._build_layer(rotary_percentage=0.25)
        inputs = ops.convert_to_tensor(
            np.random.default_rng(1).normal(size=(1, 4, 16)).astype("float32")
        )
        causal_mask = np.tril(np.ones((4, 4), dtype=bool))[None]

        full, _ = layer(
            inputs, attention_mask=ops.convert_to_tensor(causal_mask)
        )
        stepped = self._decode_step_by_step(layer, inputs, length=4)

        self.assertAllClose(full, stepped)
