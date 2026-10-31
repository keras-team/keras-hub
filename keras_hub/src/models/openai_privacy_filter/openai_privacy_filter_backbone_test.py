import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_layers import (  # noqa: E501
    OpenAIPrivacyFilterAttention,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

# Reference values for `test_small_forward_matches_reference`.
EXPECTED_FIRST_TOKEN = [
    0.170374,
    0.094855,
    0.222122,
    -0.041173,
    0.486997,
    -0.866669,
    0.894465,
    -0.603817,
]
EXPECTED_LAST_TOKEN = [
    0.086222,
    0.173202,
    -0.123395,
    0.567319,
    0.766982,
    -0.571161,
    -2.287978,
    1.340635,
]


class OpenAIPrivacyFilterBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 64,
            "intermediate_dim": 64,
            "head_dim": 16,
            "num_experts": 4,
            "top_k": 2,
            "sliding_window": 8,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 8), dtype="int32"),
            "padding_mask": ops.ones((2, 8), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=OpenAIPrivacyFilterBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 8, 64),
            # The shared precision check requires every float weight to use
            # the policy's variable dtype, but `sinks` is pinned to float32
            # to match HF. `test_precision_policies` covers the same ground
            # with that one exception.
            run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=OpenAIPrivacyFilterBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = OpenAIPrivacyFilterBackbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 189648)

    def test_small_forward_matches_reference(self):
        # Pins the model-specific numerics on a tiny config with
        # deterministic weights. Any of these one-line mutations changes the
        # values below: SwiGLU `alpha` or `limit`, the `(up + 1)` term, the
        # attention sinks column, the router softmax, the `/ top_k` and
        # `* top_k` scalings, the fp32 attention softmax, and the sliding
        # fp32 attention softmax, and the sliding window. The expected values
        # were computed once on CPU in float32 and agree across the JAX,
        # TensorFlow and PyTorch backends.
        model = OpenAIPrivacyFilterBackbone(
            vocabulary_size=16,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=8,
            head_dim=4,
            num_experts=4,
            top_k=2,
            sliding_window=2,
        )
        rng = np.random.RandomState(1234)
        model.set_weights(
            [rng.normal(scale=1.0, size=w.shape) for w in model.get_weights()]
        )
        token_ids = np.array([[3, 1, 4, 1, 5, 0]], dtype="int32")
        padding_mask = np.array([[1, 1, 1, 1, 0, 0]], dtype="int32")
        out = model({"token_ids": token_ids, "padding_mask": padding_mask})
        out = ops.convert_to_numpy(out)
        self.assertEqual(out.shape, (1, 6, 8))
        expected_first = EXPECTED_FIRST_TOKEN
        expected_last = EXPECTED_LAST_TOKEN
        atol = 2e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(out[0, 0], expected_first, atol=atol, rtol=atol)
        self.assertAllClose(out[0, 3], expected_last, atol=atol, rtol=atol)

    def test_bidirectional_attention(self):
        """Verify the model is truly bidirectional (not causal)."""
        model = OpenAIPrivacyFilterBackbone(**self.init_kwargs)
        # Input where position 0 has a different token than position 7.
        tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int32")
        mask = np.ones((1, 8), dtype="int32")
        out1 = model({"token_ids": tokens, "padding_mask": mask})

        # If attention is causal, output at pos 0 won't depend on pos 7.
        # If bidirectional, changing pos 7 should affect pos 0.
        tokens2 = np.array([[1, 2, 3, 4, 5, 6, 7, 100]], dtype="int32")
        out2 = model({"token_ids": tokens2, "padding_mask": mask})

        # In a bidirectional model, output at position 0 should change
        # when we change a token within the sliding window.
        diff = ops.convert_to_numpy(ops.sum(ops.abs(out1[0, 0] - out2[0, 0])))
        self.assertGreater(diff, 0.0)

    def test_sliding_window_mask(self):
        """Verify tokens outside the sliding window don't affect output."""
        # Use sliding_window=2 so position 0 can only see positions 0-2.
        kwargs = {**self.init_kwargs, "sliding_window": 2}
        model = OpenAIPrivacyFilterBackbone(**kwargs)

        tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int32")
        mask = np.ones((1, 8), dtype="int32")
        out1 = model({"token_ids": tokens, "padding_mask": mask})

        # Change token at position 7 (far outside window of position 0).
        tokens2 = np.array([[1, 2, 3, 4, 5, 6, 7, 100]], dtype="int32")
        out2 = model({"token_ids": tokens2, "padding_mask": mask})

        # With only 2 layers and small sliding_window=2, position 0
        # should be minimally affected by position 7 (5 hops away).
        # After 2 layers with window=2, info propagates max 4 positions.
        # So position 0 should see zero or near-zero difference.
        diff = ops.convert_to_numpy(ops.max(ops.abs(out1[0, 0] - out2[0, 0])))
        self.assertLess(diff, 1e-5)

    def test_yarn_rope_matches_hf(self):
        # Reference values from HF, with the openai/privacy-filter RoPE
        # config (the `OpenAIPrivacyFilterConfig` defaults):
        #   from transformers.models.openai_privacy_filter import (
        #       modeling_openai_privacy_filter as m,
        #   )
        #   rope = m.OpenAIPrivacyFilterRotaryEmbedding(
        #       m.OpenAIPrivacyFilterConfig()
        #   )
        #   x = torch.linspace(-1, 1, 64).reshape(1, 1, 1, 64)
        #   cos, sin = rope(x, torch.tensor([[359]]))
        #   out, _ = m.apply_rotary_pos_emb(x, x, cos, sin)
        #   out[0, 0, 0, 18:36:2], out[0, 0, 0, 19:36:2]
        # These are frequencies 9-17, the ones the YaRN ramp blends. HF
        # rotates interleaved pairs; after `_deinterleave`, frequency `i`
        # sits at channels `i` and `32 + i`.
        attention = OpenAIPrivacyFilterAttention(
            num_query_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            rope_max_wavelength=150000.0,
            rope_scaling_factor=32.0,
        )
        attention.build((None, None, 8))
        x = ops.reshape(ops.linspace(-1.0, 1.0, 64), (1, 1, 1, 64))
        outputs = attention.rotary_embedding_layer(
            attention._deinterleave(x), start_index=359
        )
        expected_even = [
            -0.712565958,
            -0.114422232,
            -0.097053885,
            0.424235344,
            0.145855650,
            -0.036326163,
            -0.051712390,
            0.010626465,
            0.099812068,
        ]
        expected_odd = [
            0.332914978,
            -0.655787706,
            0.536224484,
            0.005043730,
            -0.266478777,
            -0.180243447,
            -0.043524399,
            0.066750452,
            0.154417753,
        ]
        atol = 2e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(
            outputs[0, 0, 0, 9:18], expected_even, atol=atol, rtol=atol
        )
        self.assertAllClose(
            outputs[0, 0, 0, 41:50], expected_odd, atol=atol, rtol=atol
        )

    def test_attention_get_config_keeps_kernel_initializer(self):
        attention = OpenAIPrivacyFilterAttention(
            num_query_heads=2,
            num_key_value_heads=1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        restored = OpenAIPrivacyFilterAttention.from_config(
            attention.get_config()
        )
        restored.build((None, None, 16))
        for dense in (
            restored.query_dense,
            restored.key_dense,
            restored.value_dense,
            restored.output_dense,
        ):
            self.assertIsInstance(
                dense.kernel_initializer, keras.initializers.RandomNormal
            )
            self.assertEqual(dense.kernel_initializer.stddev, 0.02)

    def test_attention_projections_get_independent_initializers(self):
        attention = OpenAIPrivacyFilterAttention(
            num_query_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        attention.build((None, None, 16))
        self.assertNotAllClose(
            attention.query_dense.kernel, attention.key_dense.kernel
        )
        self.assertNotAllClose(
            attention.key_dense.kernel, attention.value_dense.kernel
        )

    def test_precision_policies(self):
        # Mirrors `TestCase.run_precision_test`, with one deliberate
        # exception: `sinks` stays float32 under every policy to match HF,
        # which keeps it in `_keep_in_fp32_modules_strict`.
        if keras.config.backend() == "torch":
            import torch

            if not torch.cuda.is_available():
                self.skipTest("Torch CPU has no mixed precision support.")
        for policy_name in ["mixed_float16", "mixed_bfloat16", "bfloat16"]:
            policy = keras.mixed_precision.Policy(policy_name)
            model = OpenAIPrivacyFilterBackbone(
                **{**self.init_kwargs, "dtype": policy}
            )
            output = model(self.input_data)
            self.assertDTypeEqual(output, policy.compute_dtype)
            for weight in model.weights:
                if not keras.backend.is_float_dtype(weight.dtype):
                    continue
                if weight.path.endswith("sinks"):
                    self.assertDTypeEqual(weight, "float32")
                else:
                    self.assertDTypeEqual(weight, policy.variable_dtype)

    def test_get_config_preserves_new_args(self):
        # Assert arguments from threads #15 and #16 round-trip.
        kwargs = {
            **self.init_kwargs,
            "beta_fast": 16.0,
            "beta_slow": 2.0,
            "original_max_position_embeddings": 2048,
            "truncate": True,
            "swiglu_alpha": 1.5,
            "swiglu_limit": 5.0,
        }
        model = OpenAIPrivacyFilterBackbone(**kwargs)
        restored = OpenAIPrivacyFilterBackbone.from_config(model.get_config())
        self.assertEqual(restored.beta_fast, 16.0)
        self.assertEqual(restored.beta_slow, 2.0)
        self.assertEqual(restored.original_max_position_embeddings, 2048)
        self.assertEqual(restored.truncate, True)
        self.assertEqual(restored.swiglu_alpha, 1.5)
        self.assertEqual(restored.swiglu_limit, 5.0)
