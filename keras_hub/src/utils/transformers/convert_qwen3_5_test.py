import json
import os
import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen3_5

try:
    from safetensors.numpy import save_file
except ImportError:
    save_file = None


TINY_KERAS_CONFIG = {
    "vocabulary_size": 64,
    "head_dim": 8,
    "hidden_dim": 16,
    "num_layers": 2,
    "num_query_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_dim": 24,
    "layer_norm_epsilon": 1e-6,
    "rope_max_wavelength": 10000.0,
    "partial_rotary_factor": 0.25,
    "tie_word_embeddings": True,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
}

# The same model as above, written the way a text-only HF export
# (`Qwen3_5ForCausalLM`, flat `Qwen3_5TextConfig`) writes it.
TINY_HF_TEXT_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "vocab_size": 64,
    "hidden_size": 16,
    "head_dim": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 24,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "rope_parameters": {
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "rope_type": "default",
    },
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
}


def _export_hf_text_weights(backbone, prefix):
    """Write a KerasHub Qwen3.5 backbone as HF-layout safetensors tensors.

    This is the inverse of `convert_qwen3_5.convert_weights` for the text
    decoder. `prefix` is "model" for text-only exports and
    "model.language_model" for multimodal checkpoints.
    """
    hidden = backbone.hidden_dim

    def t(x):
        return np.ascontiguousarray(np.transpose(np.asarray(x)))

    def flat_t(kernel):
        # Keras (hidden, heads, head_dim) -> HF (heads * head_dim, hidden).
        return t(np.reshape(np.asarray(kernel), (hidden, -1)))

    tensors = {}
    emb = backbone.get_layer("token_embedding")
    tensors[f"{prefix}.embed_tokens.weight"] = np.asarray(emb.embeddings)
    if not backbone.tie_word_embeddings:
        tensors["lm_head.weight"] = t(emb.reverse_embeddings)
    for i in range(backbone.num_layers):
        layer = backbone.get_layer(f"transformer_layer_{i}")
        p = f"{prefix}.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.asarray(
            layer._input_layernorm.scale
        )
        tensors[f"{p}.post_attention_layernorm.weight"] = np.asarray(
            layer._post_attention_layernorm.scale
        )
        tensors[f"{p}.mlp.up_proj.weight"] = t(
            layer._feedforward_intermediate_dense.kernel
        )
        tensors[f"{p}.mlp.down_proj.weight"] = t(
            layer._feedforward_output_dense.kernel
        )
        tensors[f"{p}.mlp.gate_proj.weight"] = t(
            layer._feedforward_gate_dense.kernel
        )
        if layer.layer_type == "full_attention":
            attn = layer._self_attention_layer
            tensors[f"{p}.self_attn.q_proj.weight"] = flat_t(
                attn._query_dense.kernel
            )
            tensors[f"{p}.self_attn.k_proj.weight"] = flat_t(
                attn._key_dense.kernel
            )
            tensors[f"{p}.self_attn.v_proj.weight"] = flat_t(
                attn._value_dense.kernel
            )
            # Keras (heads, head_dim, hidden) -> HF (hidden, heads * head_dim)
            tensors[f"{p}.self_attn.o_proj.weight"] = t(
                np.reshape(np.asarray(attn._output_dense.kernel), (-1, hidden))
            )
            tensors[f"{p}.self_attn.q_norm.weight"] = np.asarray(
                attn._query_norm.scale
            )
            tensors[f"{p}.self_attn.k_norm.weight"] = np.asarray(
                attn._key_norm.scale
            )
        else:
            gdn = layer._linear_attn
            tensors[f"{p}.linear_attn.in_proj_qkv.weight"] = t(
                gdn.in_proj_qkv.kernel
            )
            tensors[f"{p}.linear_attn.in_proj_z.weight"] = t(
                gdn.in_proj_z.kernel
            )
            tensors[f"{p}.linear_attn.in_proj_b.weight"] = t(
                gdn.in_proj_b.kernel
            )
            tensors[f"{p}.linear_attn.in_proj_a.weight"] = t(
                gdn.in_proj_a.kernel
            )
            # Keras (channels, kernel) -> HF (channels, 1, kernel).
            tensors[f"{p}.linear_attn.conv1d.weight"] = np.expand_dims(
                np.asarray(gdn.conv1d_weight), 1
            )
            tensors[f"{p}.linear_attn.dt_bias"] = np.asarray(gdn.dt_bias)
            tensors[f"{p}.linear_attn.A_log"] = np.asarray(gdn.A_log)
            tensors[f"{p}.linear_attn.norm.weight"] = np.asarray(gdn.norm.scale)
            tensors[f"{p}.linear_attn.out_proj.weight"] = t(gdn.out_proj.kernel)
    tensors[f"{prefix}.norm.weight"] = np.asarray(
        backbone.get_layer("sequence_output_layernorm").scale
    )
    return {
        k: np.ascontiguousarray(v.astype("float32")) for k, v in tensors.items()
    }


def _write_hf_preset(preset_dir, hf_config, tensors, sharded=False):
    with open(os.path.join(preset_dir, "config.json"), "w") as f:
        json.dump(hf_config, f)
    if not sharded:
        save_file(tensors, os.path.join(preset_dir, "model.safetensors"))
        return
    keys = sorted(tensors)
    half = len(keys) // 2
    shards = {
        "model-00001-of-00002.safetensors": keys[:half],
        "model-00002-of-00002.safetensors": keys[half:],
    }
    weight_map = {}
    for fname, shard_keys in shards.items():
        save_file(
            {k: tensors[k] for k in shard_keys},
            os.path.join(preset_dir, fname),
        )
        weight_map.update({k: fname for k in shard_keys})
    index = {"metadata": {}, "weight_map": weight_map}
    with open(
        os.path.join(preset_dir, "model.safetensors.index.json"), "w"
    ) as f:
        json.dump(index, f)


class TestQwen3_5Converter(TestCase):
    def _make_random_backbone(self, **overrides):
        backbone = Qwen3_5Backbone(**{**TINY_KERAS_CONFIG, **overrides})
        rng = np.random.default_rng(0)
        for weight in backbone.weights:
            weight.assign(
                rng.uniform(-0.5, 0.5, size=weight.shape).astype("float32")
            )
        return backbone

    def _assert_roundtrip(self, hf_config, prefix, sharded=False, **overrides):
        """Export a random backbone in HF layout, reload it, compare."""
        if save_file is None:
            self.skipTest("safetensors is not installed.")
        backbone = self._make_random_backbone(**overrides)
        tensors = _export_hf_text_weights(backbone, prefix)
        token_ids = np.array([[1, 5, 9, 3]], dtype="int32")
        inputs = {
            "token_ids": token_ids,
            "padding_mask": np.ones_like(token_ids),
        }
        expected = ops.convert_to_numpy(backbone(inputs))

        with tempfile.TemporaryDirectory() as preset_dir:
            _write_hf_preset(preset_dir, hf_config, tensors, sharded=sharded)
            loaded = Backbone.from_preset(preset_dir)

        self.assertIsInstance(loaded, Qwen3_5Backbone)
        self.assertIsNone(loaded.vision_encoder)
        self.assertEqual(
            loaded.tie_word_embeddings, backbone.tie_word_embeddings
        )
        actual = ops.convert_to_numpy(loaded(inputs))
        self.assertAllClose(actual, expected, atol=1e-6)

    def test_text_only_export_single_file(self):
        # Regression test for keras-team/keras-hub#3173: text-only
        # `Qwen3_5ForCausalLM` exports use model_type `qwen3_5_text` and
        # `model.layers.*` keys rather than `model.language_model.layers.*`.
        self._assert_roundtrip(TINY_HF_TEXT_CONFIG, prefix="model")

    def test_text_only_export_sharded(self):
        self._assert_roundtrip(
            TINY_HF_TEXT_CONFIG, prefix="model", sharded=True
        )

    def test_text_only_export_untied_lm_head(self):
        hf_config = {**TINY_HF_TEXT_CONFIG, "tie_word_embeddings": False}
        self._assert_roundtrip(
            hf_config, prefix="model", tie_word_embeddings=False
        )

    def test_multimodal_layout_still_loads(self):
        # A multimodal-style config (nested `text_config`, model_type
        # `qwen3_5`) with the decoder under `model.language_model.*`.
        text_keys = [
            k
            for k in TINY_HF_TEXT_CONFIG
            if k not in ("architectures", "model_type", "tie_word_embeddings")
        ]
        hf_config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": True,
            "text_config": {k: TINY_HF_TEXT_CONFIG[k] for k in text_keys},
        }
        self._assert_roundtrip(hf_config, prefix="model.language_model")
        self._assert_roundtrip(
            hf_config, prefix="model.language_model", sharded=True
        )

    def test_is_text_only_config(self):
        self.assertTrue(
            convert_qwen3_5.is_text_only_config(TINY_HF_TEXT_CONFIG)
        )
        self.assertTrue(
            convert_qwen3_5.is_text_only_config({"model_type": "qwen3_5"})
        )
        self.assertFalse(
            convert_qwen3_5.is_text_only_config(
                {"model_type": "qwen3_5", "vision_config": {}}
            )
        )

    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3_5Backbone.from_preset(
            "hf://yujiepan/qwen3.5-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5Backbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen3.5-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5Backbone)

    def test_qwen3_5_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "head_dim": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"partial_rotary_factor": 0.5},
            "tie_word_embeddings": True,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
        }
        keras_config = convert_qwen3_5.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {
            "rope_theta": 20000.0,
            "partial_rotary_factor": 0.5,
        }
        keras_config = convert_qwen3_5.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
