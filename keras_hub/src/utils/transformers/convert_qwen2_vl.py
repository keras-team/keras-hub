"""Convert HuggingFace Qwen2-VL weights to KerasHub."""

import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen2VLBackbone


def convert_backbone_config(transformers_config):
    # Newer transformers nest text params under "text_config".
    tc = transformers_config.get("text_config", transformers_config)
    vision_config = transformers_config.get("vision_config", {})
    rope_params = tc.get("rope_parameters", {})
    rope_theta = tc.get("rope_theta", rope_params.get("rope_theta", 1000000))
    return {
        "vocabulary_size": tc["vocab_size"],
        "num_layers": tc["num_hidden_layers"],
        "num_query_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "hidden_dim": tc["hidden_size"],
        "intermediate_dim": tc["intermediate_size"],
        "vision_patch_size": vision_config.get("patch_size", 14),
        "vision_temporal_patch_size": vision_config.get(
            "temporal_patch_size", 2
        ),
        "vision_in_channels": vision_config.get("in_channels", 3),
        "vision_embed_dim": vision_config.get(
            "embed_dim", vision_config.get("hidden_size", 1280)
        ),
        "vision_depth": vision_config.get(
            "depth", vision_config.get("num_hidden_layers", 32)
        ),
        "vision_num_heads": vision_config.get(
            "num_heads", vision_config.get("num_attention_heads", 16)
        ),
        "vision_mlp_ratio": vision_config.get("mlp_ratio", 4),
        "spatial_merge_size": vision_config.get("spatial_merge_size", 2),
        "image_token_id": transformers_config.get("image_token_id", 151655),
        "rope_max_wavelength": rope_theta,
        "layer_norm_epsilon": tc.get("rms_norm_eps", 1e-6),
        "tie_word_embeddings": transformers_config.get(
            "tie_word_embeddings", False
        ),
        "use_sliding_window_attention": tc.get("use_sliding_window", False),
        "sliding_window_size": tc.get("sliding_window", 32768),
    }


def convert_weights(backbone, loader, transformers_config):
    # ── helpers ──────────────────────────────────────────────────────
    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    def transpose_2d(x, _):
        return np.transpose(x, axes=(1, 0))

    # ── token embeddings ────────────────────────────────────────────
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="lm_head.weight",
            hook_fn=transpose_2d,
        )

    # ── text decoder layers ─────────────────────────────────────────
    for i in range(backbone.num_layers):
        decoder = backbone.get_layer(f"transformer_layer_{i}")

        # Pre-attention RMSNorm
        loader.port_weight(
            keras_variable=decoder._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Q/K/V projections (EinsumDense → transpose + reshape)
        loader.port_weight(
            keras_variable=(decoder._self_attention_layer._query_dense.kernel),
            hf_weight_key=(f"model.layers.{i}.self_attn.q_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder._self_attention_layer._query_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.bias",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=(decoder._self_attention_layer._key_dense.kernel),
            hf_weight_key=(f"model.layers.{i}.self_attn.k_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder._self_attention_layer._key_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.bias",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=(decoder._self_attention_layer._value_dense.kernel),
            hf_weight_key=(f"model.layers.{i}.self_attn.v_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder._self_attention_layer._value_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.bias",
            hook_fn=transpose_and_reshape,
        )

        # Output projection
        loader.port_weight(
            keras_variable=(decoder._self_attention_layer._output_dense.kernel),
            hf_weight_key=(f"model.layers.{i}.self_attn.o_proj.weight"),
            hook_fn=transpose_and_reshape,
        )

        # MLP (gate / up / down)
        loader.port_weight(
            keras_variable=decoder._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=decoder._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=decoder._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            hook_fn=transpose_2d,
        )

        # Post-attention RMSNorm
        loader.port_weight(
            keras_variable=decoder._feedforward_layernorm.scale,
            hf_weight_key=(f"model.layers.{i}.post_attention_layernorm.weight"),
        )

    # Final layernorm
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    # ── vision encoder ──────────────────────────────────────────────
    vision = backbone.get_layer("vision_encoder")

    # Patch embedding (Conv3D)
    # HF: (embed_dim, C, T, H, W) → Keras: (T, H, W, C, embed_dim)
    loader.port_weight(
        keras_variable=vision.patch_embed.kernel,
        hf_weight_key="visual.patch_embed.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    # HF patch_embed has bias, KerasHub does not — skip it.

    # Vision blocks
    for i in range(vision.depth):
        block = vision.blocks[i]
        prefix = f"visual.blocks.{i}"

        # LayerNorm 1
        loader.port_weight(
            keras_variable=block.norm1.gamma,
            hf_weight_key=f"{prefix}.norm1.weight",
        )
        loader.port_weight(
            keras_variable=block.norm1.beta,
            hf_weight_key=f"{prefix}.norm1.bias",
        )

        # Fused QKV attention
        loader.port_weight(
            keras_variable=block.attn.qkv.kernel,
            hf_weight_key=f"{prefix}.attn.qkv.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.attn.qkv.bias,
            hf_weight_key=f"{prefix}.attn.qkv.bias",
        )

        # Output projection
        loader.port_weight(
            keras_variable=block.attn.proj.kernel,
            hf_weight_key=f"{prefix}.attn.proj.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.attn.proj.bias,
            hf_weight_key=f"{prefix}.attn.proj.bias",
        )

        # LayerNorm 2
        loader.port_weight(
            keras_variable=block.norm2.gamma,
            hf_weight_key=f"{prefix}.norm2.weight",
        )
        loader.port_weight(
            keras_variable=block.norm2.beta,
            hf_weight_key=f"{prefix}.norm2.bias",
        )

        # MLP
        loader.port_weight(
            keras_variable=block.mlp.fc1.kernel,
            hf_weight_key=f"{prefix}.mlp.fc1.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.mlp.fc1.bias,
            hf_weight_key=f"{prefix}.mlp.fc1.bias",
        )
        loader.port_weight(
            keras_variable=block.mlp.fc2.kernel,
            hf_weight_key=f"{prefix}.mlp.fc2.weight",
            hook_fn=transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.mlp.fc2.bias,
            hf_weight_key=f"{prefix}.mlp.fc2.bias",
        )

    # Patch merger
    merger = vision.merger
    loader.port_weight(
        keras_variable=merger.ln_q.gamma,
        hf_weight_key="visual.merger.ln_q.weight",
    )
    loader.port_weight(
        keras_variable=merger.ln_q.beta,
        hf_weight_key="visual.merger.ln_q.bias",
    )
    # HF merger MLP is nn.Sequential(Linear, GELU, Linear)
    # sub-indices: .0 = fc1, .1 = GELU (no weights), .2 = fc2
    loader.port_weight(
        keras_variable=merger.mlp_fc1.kernel,
        hf_weight_key="visual.merger.mlp.0.weight",
        hook_fn=transpose_2d,
    )
    loader.port_weight(
        keras_variable=merger.mlp_fc1.bias,
        hf_weight_key="visual.merger.mlp.0.bias",
    )
    loader.port_weight(
        keras_variable=merger.mlp_fc2.kernel,
        hf_weight_key="visual.merger.mlp.2.weight",
        hook_fn=transpose_2d,
    )
    loader.port_weight(
        keras_variable=merger.mlp_fc2.bias,
        hf_weight_key="visual.merger.mlp.2.bias",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    # Load all special tokens except reserved placeholders.
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
