import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_decoder import (
    Qwen3_VLTransformerDecoder,
)
from keras_hub.src.tests.test_case import TestCase

HIDDEN = 32
HEAD_DIM = 16
SECTION = [4, 2, 2]

try:
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as modeling
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
except ImportError:
    torch = None


def _decoder(**kwargs):
    config = dict(
        intermediate_dim=48,
        head_dim=HEAD_DIM,
        num_query_heads=4,
        num_key_value_heads=2,
        rope_max_wavelength=10000.0,
        mrope_section=SECTION,
        layer_norm_epsilon=1e-6,
    )
    config.update(kwargs)
    return Qwen3_VLTransformerDecoder(**config)


class Qwen3VLDecoderTest(TestCase):
    def setUp(self):
        self.x = np.random.RandomState(0).normal(size=(2, 6, HIDDEN))
        self.x = self.x.astype("float32")
        rng = np.random.RandomState(1)
        self.position_ids = np.sort(
            rng.randint(0, 40, size=(2, 3, 6)), axis=-1
        ).astype("int32")

    def test_layer_basics(self):
        self.run_layer_test(
            cls=Qwen3_VLTransformerDecoder,
            init_kwargs=dict(
                intermediate_dim=48,
                head_dim=HEAD_DIM,
                num_query_heads=4,
                num_key_value_heads=2,
                rope_max_wavelength=10000.0,
                mrope_section=SECTION,
                layer_norm_epsilon=1e-6,
                activation="silu",
                kernel_initializer="glorot_uniform",
                dropout=0.0,
            ),
            input_data=self.x,
            expected_output_shape=(2, 6, HIDDEN),
            expected_num_trainable_weights=11,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_get_config_round_trip_non_default(self):
        layer = _decoder(
            activation="gelu",
            dropout=0.2,
            rope_max_wavelength=123.0,
            layer_norm_epsilon=1e-5,
        )
        self.run_serialization_test(layer)

    def test_padding_mask(self):
        layer = _decoder()
        mask = np.ones((2, 6), dtype="int32")
        mask[1, 4:] = 0
        out = ops.convert_to_numpy(
            layer(
                self.x,
                decoder_padding_mask=mask,
                position_ids=self.position_ids,
            )
        )
        # Changing masked-out inputs leaves earlier outputs unchanged.
        x2 = self.x.copy()
        x2[1, 4:] += 3.0
        out2 = ops.convert_to_numpy(
            layer(x2, decoder_padding_mask=mask, position_ids=self.position_ids)
        )
        self.assertAllClose(out[1, :4], out2[1, :4], atol=1e-5)

    def test_cache_matches_no_cache(self):
        layer = _decoder()
        full = ops.convert_to_numpy(
            layer(self.x, position_ids=self.position_ids)
        )
        cache = np.zeros((2, 2, 6, 2, HEAD_DIM), dtype="float32")
        # Prefill 4 tokens, then decode the rest one by one.
        out, cache = layer(
            self.x[:, :4],
            position_ids=self.position_ids[:, :, :4],
            cache=cache,
            cache_update_index=0,
        )
        outputs = [ops.convert_to_numpy(out)]
        for i in (4, 5):
            out, cache = layer(
                self.x[:, i : i + 1],
                position_ids=self.position_ids[:, :, i : i + 1],
                cache=cache,
                cache_update_index=i,
            )
            outputs.append(ops.convert_to_numpy(out))
        self.assertAllClose(np.concatenate(outputs, axis=1), full, atol=1e-5)

    @pytest.mark.skipif(
        torch is None, reason="Requires torch and transformers."
    )
    def test_matches_hf_text_decoder_layer(self):
        config = Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=HIDDEN,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=HEAD_DIM,
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": SECTION,
                "mrope_interleaved": True,
            },
        )
        config._attn_implementation = "eager"
        torch.manual_seed(0)
        hf_layer = modeling.Qwen3VLTextDecoderLayer(config, 0).eval()
        with torch.no_grad():
            for p in hf_layer.parameters():
                p.copy_(torch.randn_like(p) * 0.2)
        rotary = modeling.Qwen3VLTextRotaryEmbedding(config)

        x = torch.tensor(self.x)
        pos = torch.tensor(self.position_ids, dtype=torch.long)
        cos_sin = rotary(x, pos.permute(1, 0, 2))
        causal = torch.full((6, 6), float("-inf")).triu(1)[None, None]
        with torch.no_grad():
            expected = hf_layer(
                x, position_embeddings=cos_sin, attention_mask=causal
            ).numpy()

        layer = _decoder()
        layer.build(self.x.shape)
        attn = hf_layer.self_attn
        mlp = hf_layer.mlp
        ours = layer._self_attention_layer

        def np_(t):
            return t.detach().numpy()

        def split_heads(w, heads):
            return np_(w).T.reshape(HIDDEN, heads, HEAD_DIM)

        ours._query_dense.kernel.assign(split_heads(attn.q_proj.weight, 4))
        ours._key_dense.kernel.assign(split_heads(attn.k_proj.weight, 2))
        ours._value_dense.kernel.assign(split_heads(attn.v_proj.weight, 2))
        ours._output_dense.kernel.assign(
            np_(attn.o_proj.weight).T.reshape(4, HEAD_DIM, HIDDEN)
        )
        ours._query_norm.scale.assign(np_(attn.q_norm.weight))
        ours._key_norm.scale.assign(np_(attn.k_norm.weight))
        layer._feedforward_gate_dense.kernel.assign(np_(mlp.gate_proj.weight).T)
        layer._feedforward_intermediate_dense.kernel.assign(
            np_(mlp.up_proj.weight).T
        )
        layer._feedforward_output_dense.kernel.assign(
            np_(mlp.down_proj.weight).T
        )
        layer._self_attention_layernorm.scale.assign(
            np_(hf_layer.input_layernorm.weight)
        )
        layer._feedforward_layernorm.scale.assign(
            np_(hf_layer.post_attention_layernorm.weight)
        )

        actual = ops.convert_to_numpy(
            layer(self.x, position_ids=self.position_ids)
        )
        self.assertLess(np.abs(actual - expected).max(), 1e-4)
