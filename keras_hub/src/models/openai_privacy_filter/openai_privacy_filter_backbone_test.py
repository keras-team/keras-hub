import pytest
from keras import ops

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterAttention,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.tests.test_case import TestCase


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
            run_quantization_check=False,
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
        # Verify the model can count its own params.
        self.assertGreater(model.count_params(), 0)

    def test_bidirectional_attention(self):
        """Verify the model is truly bidirectional (not causal)."""
        import numpy as np

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
        import numpy as np

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
        self.assertAllClose(outputs[0, 0, 0, 9:18], expected_even, atol=1e-4)
        self.assertAllClose(outputs[0, 0, 0, 41:50], expected_odd, atol=1e-4)
