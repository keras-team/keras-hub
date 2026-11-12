import numpy as np
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_attention import (
    UnlimitedOCRAttention,
)
from keras_hub.src.tests.test_case import TestCase


class UnlimitedOCRAttentionTest(TestCase):
    def setUp(self):
        self.init_kwargs = {"num_query_heads": 2}
        self.input_data = np.random.uniform(size=(2, 6, 16)).astype("float32")

    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRAttention,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 6, 16),
            # The query, key, value and output kernels. No layer has a bias.
            expected_num_trainable_weights=4,
        )

    def test_attention_mask_blocks_positions(self):
        layer = UnlimitedOCRAttention(**self.init_kwargs)
        x = self.input_data
        causal = np.tril(np.ones((6, 6), dtype="bool"))
        mask = np.broadcast_to(causal, (2, 6, 6))
        masked = ops.convert_to_numpy(layer(x, attention_mask=mask))
        changed = x.copy()
        changed[:, 5] += 1.0
        masked_changed = ops.convert_to_numpy(
            layer(changed, attention_mask=mask)
        )
        # A causal mask keeps earlier positions independent of later tokens.
        self.assertAllClose(masked[:, :5], masked_changed[:, :5], atol=1e-5)
        unmasked = ops.convert_to_numpy(layer(x))
        unmasked_changed = ops.convert_to_numpy(layer(changed))
        difference = np.abs(unmasked[:, 0] - unmasked_changed[:, 0]).max()
        self.assertGreater(float(difference), 1e-6)

    def test_cached_decode_matches_full_pass(self):
        layer = UnlimitedOCRAttention(**self.init_kwargs)
        seq_len = 6
        x = self.input_data
        causal = np.tril(np.ones((seq_len, seq_len), dtype="bool"))
        expected = ops.convert_to_numpy(
            layer(x, attention_mask=np.broadcast_to(causal, (2, 6, 6)))
        )

        cache = ops.zeros((2, 2, seq_len, 2, 8))
        outputs = []
        for index in range(seq_len):
            # The query at `index` sees the keys up to `index`.
            mask = np.zeros((2, 1, seq_len), dtype="bool")
            mask[:, :, : index + 1] = True
            step, cache = layer(
                x[:, index : index + 1],
                attention_mask=mask,
                cache=cache,
                cache_update_index=index,
            )
            outputs.append(ops.convert_to_numpy(step))
        actual = np.concatenate(outputs, axis=1)
        self.assertAllClose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_cache_update_index_needs_a_cache(self):
        layer = UnlimitedOCRAttention(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "cache_update_index"):
            layer(self.input_data, cache_update_index=0)
