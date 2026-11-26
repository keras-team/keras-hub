import keras
import numpy as np
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_decoder import (
    UnlimitedOCRTransformerDecoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


def _tolerance(atol, rtol, gpu_atol, gpu_rtol):
    """Return the tolerances. GPU matmuls use TF32 and need larger ones."""
    if running_on_gpu():
        return {"atol": gpu_atol, "rtol": gpu_rtol}
    return {"atol": atol, "rtol": rtol}


class UnlimitedOCRDecoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "intermediate_dim": 32,
            "num_query_heads": 2,
            "moe_intermediate_dim": 8,
            "num_shared_experts": 2,
            "num_experts": 4,
            "top_k": 2,
            "sliding_window": 3,
        }
        self.input_data = np.random.uniform(size=(2, 6, 16)).astype("float32")

    def test_sparse_decoder_layer(self):
        self.run_layer_test(
            cls=UnlimitedOCRTransformerDecoder,
            init_kwargs={**self.init_kwargs, "sparse": True},
            input_data=self.input_data,
            expected_output_shape=(2, 6, 16),
            # Attention (4), two norms (2), router (1), experts (2) and the
            # shared experts (3).
            expected_num_trainable_weights=12,
        )

    def test_dense_decoder_layer(self):
        self.run_layer_test(
            cls=UnlimitedOCRTransformerDecoder,
            init_kwargs={**self.init_kwargs, "sparse": False},
            input_data=self.input_data,
            expected_output_shape=(2, 6, 16),
            # Attention (4), two norms (2) and the dense MLP (3).
            expected_num_trainable_weights=9,
        )

    def test_reference_sliding_window_mask(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        # One decode query at position 5 over a cache of length 8. The
        # window is 3, so the query sees positions 3, 4 and 5.
        sequence = np.zeros((2, 1, 16), dtype="float32")
        cache = np.zeros((2, 2, 8, 2, 8), dtype="float32")
        mask = layer._compute_self_attention_mask(
            decoder_sequence=ops.convert_to_tensor(sequence),
            decoder_padding_mask=None,
            decoder_attention_mask=None,
            self_attention_cache=ops.convert_to_tensor(cache),
            self_attention_cache_update_index=5,
            prefill_length=ops.convert_to_tensor([2, 4], dtype="int32"),
        )
        mask = ops.convert_to_numpy(mask)
        self.assertEqual(mask.shape, (2, 1, 8))
        expected = np.zeros((2, 1, 8), dtype="bool")
        expected[0, 0, [0, 1, 3, 4, 5]] = True
        expected[1, 0, [0, 1, 2, 3, 4, 5]] = True
        self.assertAllEqual(mask, expected)

    def test_no_sliding_window_gives_full_causal_mask(self):
        layer = UnlimitedOCRTransformerDecoder(
            **{**self.init_kwargs, "sliding_window": None}
        )
        # The same call as the R-SWA mask test. Without a window, a query at
        # position 5 sees every position up to 5 in both rows.
        mask = layer._compute_self_attention_mask(
            decoder_sequence=ops.convert_to_tensor(
                np.zeros((2, 1, 16), dtype="float32")
            ),
            decoder_padding_mask=None,
            decoder_attention_mask=None,
            self_attention_cache=ops.convert_to_tensor(
                np.zeros((2, 2, 8, 2, 8), dtype="float32")
            ),
            self_attention_cache_update_index=5,
            prefill_length=ops.convert_to_tensor([2, 4], dtype="int32"),
        )
        expected = np.zeros((2, 1, 8), dtype="bool")
        expected[:, 0, :6] = True
        self.assertAllEqual(ops.convert_to_numpy(mask), expected)

    def test_cached_decode_matches_reference_mask(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        seq_len, prefill = 8, np.array([4, 3], dtype="int32")
        window = self.init_kwargs["sliding_window"]
        x = np.random.normal(size=(2, seq_len, 16)).astype("float32")
        # Build the reference mask with NumPy. It combines causal attention
        # with the prefill tokens and the sliding window.
        query = np.arange(seq_len)[:, None]
        key = np.arange(seq_len)[None, :]
        causal = key <= query
        reference = np.zeros((2, seq_len, seq_len), dtype="bool")
        for row in range(2):
            allowed = (key < prefill[row]) | (key > query - window)
            reference[row] = causal & allowed
        expected = ops.convert_to_numpy(
            layer(x, decoder_attention_mask=ops.convert_to_tensor(reference))
        )

        # Run the same sequence one token at a time through the cache.
        head_dim = 16 // self.init_kwargs["num_query_heads"]
        cache = ops.zeros(
            (2, 2, seq_len, self.init_kwargs["num_query_heads"], head_dim)
        )
        outputs = []
        for index in range(seq_len):
            step, cache = layer(
                x[:, index : index + 1],
                self_attention_cache=cache,
                self_attention_cache_update_index=index,
                prefill_length=ops.convert_to_tensor(prefill),
            )
            outputs.append(ops.convert_to_numpy(step))
        actual = np.concatenate(outputs, axis=1)
        self.assertAllClose(
            actual, expected, **_tolerance(1e-4, 1e-4, 1e-2, 1e-2)
        )

    def test_sliding_window_changes_output(self):
        # With a short window and a long decode, the R-SWA output must differ
        # from full causal attention. This guards a mask that never applies.
        # Both runs decode through the cache, so only the mask differs.
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        seq_len = 8
        x = np.random.normal(size=(1, seq_len, 16)).astype("float32")
        prefill = ops.convert_to_tensor([2], dtype="int32")

        def decode(prefill_length):
            cache = ops.zeros((1, 2, seq_len, 2, 8))
            outputs = []
            for index in range(seq_len):
                step, cache = layer(
                    x[:, index : index + 1],
                    self_attention_cache=cache,
                    self_attention_cache_update_index=index,
                    prefill_length=prefill_length,
                )
                outputs.append(ops.convert_to_numpy(step))
            return np.concatenate(outputs, axis=1)

        windowed = decode(prefill)
        full = decode(None)
        # Positions before the window fills match full attention.
        self.assertAllClose(
            windowed[:, :3], full[:, :3], **_tolerance(1e-4, 1e-4, 1e-2, 1e-2)
        )
        difference = np.abs(windowed[:, 7] - full[:, 7]).max()
        self.assertGreater(float(difference), 1e-2)

    def test_compute_output_shape(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        self.assertEqual(layer.compute_output_shape((2, 6, 16)), (2, 6, 16))

    def test_compute_output_shape_with_cache(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        output_shape, cache_shape = layer.compute_output_shape(
            (2, 6, 16), self_attention_cache_shape=(2, 2, 8, 2, 8)
        )
        self.assertEqual(output_shape, (2, 6, 16))
        self.assertEqual(cache_shape, (2, 2, 8, 2, 8))

    def test_compute_output_spec(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        cache = np.zeros((2, 2, 8, 2, 8), dtype="float32")
        # NumPy, backend tensor and symbolic inputs give the same spec.
        for x in (
            self.input_data,
            ops.convert_to_tensor(self.input_data),
            keras.KerasTensor((2, 6, 16)),
        ):
            spec = layer.compute_output_spec(x)
            self.assertEqual(spec.shape, (2, 6, 16))
            output_spec, cache_spec = layer.compute_output_spec(
                x,
                self_attention_cache=cache,
                self_attention_cache_update_index=0,
                prefill_length=np.array([2, 4], dtype="int32"),
            )
            self.assertEqual(output_spec.shape, (2, 6, 16))
            self.assertEqual(cache_spec.shape, (2, 2, 8, 2, 8))

    def test_functional_model_with_cache(self):
        layer = UnlimitedOCRTransformerDecoder(**self.init_kwargs)
        inputs = keras.Input(shape=(6, 16))
        cache = keras.Input(shape=(2, 8, 2, 8))
        outputs = layer(
            inputs,
            self_attention_cache=cache,
            self_attention_cache_update_index=0,
        )
        model = keras.Model([inputs, cache], outputs)
        output, new_cache = model.predict(
            [self.input_data, np.zeros((2, 2, 8, 2, 8), dtype="float32")],
            verbose=0,
        )
        self.assertEqual(output.shape, (2, 6, 16))
        self.assertEqual(new_cache.shape, (2, 2, 8, 2, 8))
