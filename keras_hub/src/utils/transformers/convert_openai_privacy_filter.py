"""OpenAI Privacy Filter checkpoint conversion."""

import json

import numpy as np

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = OpenAIPrivacyFilterBackbone


def convert_backbone_config(transformers_config):
    rope_params = transformers_config.get("rope_parameters", {})
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "head_dim": transformers_config.get("head_dim", 64),
        "num_experts": transformers_config["num_local_experts"],
        "top_k": transformers_config["num_experts_per_tok"],
        "rope_max_wavelength": rope_params.get(
            "rope_theta",
            transformers_config.get("rope_theta", 150000.0),
        ),
        "rope_scaling_factor": rope_params.get("factor", 32.0),
        "rms_norm_eps": transformers_config.get("rms_norm_eps", 1e-5),
        "sliding_window": transformers_config.get("sliding_window", 128),
        "attention_dropout": transformers_config.get("attention_dropout", 0.0),
    }


def convert_weights(backbone, loader, transformers_config):
    # Token embedding
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )

    for i in range(backbone.num_layers):
        layer = backbone.encoder_layers[i]
        attn = layer.self_attention

        # Input layernorm
        loader.port_weight(
            keras_variable=layer.input_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Q/K/V/O projections
        for proj_name, keras_dense in [
            ("q_proj", attn.query_dense),
            ("k_proj", attn.key_dense),
            ("v_proj", attn.value_dense),
            ("o_proj", attn.output_dense),
        ]:
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=(
                    f"model.layers.{i}.self_attn.{proj_name}.weight"
                ),
                hook_fn=lambda hf_tensor, shape: np.reshape(
                    np.transpose(hf_tensor, axes=(1, 0)), shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_dense.bias,
                hf_weight_key=(f"model.layers.{i}.self_attn.{proj_name}.bias"),
                hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
            )

        # Sink tokens
        loader.port_weight(
            keras_variable=attn.sinks,
            hf_weight_key=f"model.layers.{i}.self_attn.sinks",
        )

        # Post-attention layernorm
        loader.port_weight(
            keras_variable=layer.post_attention_layernorm.scale,
            hf_weight_key=(f"model.layers.{i}.post_attention_layernorm.weight"),
        )

        # MoE router
        moe = layer.sparse_moe_block
        loader.port_weight(
            keras_variable=moe.router.router_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.router.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=moe.router.router_dense.bias,
            hf_weight_key=f"model.layers.{i}.mlp.router.bias",
        )

        # Expert weights — direct safetensor weights (not MXFP4)
        # HF stores as (num_experts, hidden_dim, 2*intermediate_dim) which
        # matches our model layout directly — no transpose needed.
        loader.port_weight(
            keras_variable=moe.experts.gate_up_proj,
            hf_weight_key=(f"model.layers.{i}.mlp.experts.gate_up_proj"),
        )
        loader.port_weight(
            keras_variable=moe.experts.gate_up_proj_bias,
            hf_weight_key=(f"model.layers.{i}.mlp.experts.gate_up_proj_bias"),
        )
        loader.port_weight(
            keras_variable=moe.experts.down_proj,
            hf_weight_key=(f"model.layers.{i}.mlp.experts.down_proj"),
        )
        loader.port_weight(
            keras_variable=moe.experts.down_proj_bias,
            hf_weight_key=(f"model.layers.{i}.mlp.experts.down_proj_bias"),
        )

    # Final layernorm
    loader.port_weight(
        keras_variable=backbone.final_layernorm.scale,
        hf_weight_key="model.norm.weight",
    )


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_file = get_file(preset, "tokenizer.json")
    with open(tokenizer_file, "r") as f:
        tokenizer_data = json.load(f)
    vocabulary = tokenizer_data.get("model", {}).get("vocab", {})
    merges = tokenizer_data.get("model", {}).get("merges", [])
    added_tokens = tokenizer_data.get("added_tokens", [])

    vocab_dict = {}
    for token, token_id in vocabulary.items():
        vocab_dict[token] = int(token_id)
    for token_info in added_tokens:
        vocab_dict[token_info["content"]] = int(token_info["id"])

    merges_strings = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merges_strings.append(f"{merge[0]} {merge[1]}")
        else:
            merges_strings.append(str(merge))

    return cls(vocabulary=vocab_dict, merges=merges_strings, **kwargs)


def convert_head(task, loader, transformers_config):
    """Load classification head weights."""
    loader.port_weight(
        keras_variable=task._classifier_dense.kernel,
        hf_weight_key="score.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=task._classifier_dense.bias,
        hf_weight_key="score.bias",
    )
