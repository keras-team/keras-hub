"""Convert HuggingFace Qwen3Model (embedding) checkpoints to KerasHub.

This converter handles Qwen3-based text embedding models such as
Microsoft's harrier-oss family. These checkpoints declare
`"architectures": ["Qwen3Model"]` in their config rather than the
CausalLM variant, and some (e.g. harrier) omit the standard ``model.``
weight-key prefix used by Qwen3ForCausalLM checkpoints.

The converter is registered in ``preset_loader.py`` for
``model_type == "qwen3"`` when the ``architectures`` field contains
``"Qwen3Model"``.  The plain ``convert_qwen3`` converter handles all
other ``model_type == "qwen3"`` checkpoints (CausalLM).
"""

import numpy as np

from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3Backbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "head_dim": transformers_config["head_dim"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "rope_max_wavelength": transformers_config["rope_theta"],
        "sliding_window_size": transformers_config["sliding_window"]
        if transformers_config["use_sliding_window"]
        else None,
        "tie_word_embeddings": transformers_config["tie_word_embeddings"],
    }


def convert_weights(backbone, loader, transformers_config):
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"layers.{i}.input_layernorm.weight",
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense_layer_norm.scale,
            hf_weight_key=f"layers.{i}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense_layer_norm.scale,
            hf_weight_key=f"layers.{i}.self_attn.k_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    merges = [" ".join(item) for item in merges]

    # Load all special tokens with the exception of "reserved" ones.
    special_tokens = set()
    for token in tokenizer_config["added_tokens"]:
        if not token["content"].startswith("<|reserved_special_token_"):
            vocab[token["content"]] = token["id"]
            special_tokens.add(token["content"])

    kwargs.update(
        {
            "unsplittable_tokens": list(special_tokens),
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)
