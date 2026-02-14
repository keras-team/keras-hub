"""Convert Qwen2-VL weights from HuggingFace Transformers."""

import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen2VLBackbone


def convert_backbone_config(transformers_config):
    from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
        Qwen2VLVisionEncoder,
    )

    vision_config = transformers_config.get("vision_config", {})
    mrope_section = transformers_config.get("rope_scaling", {}).get(
        "mrope_section", [16, 24, 24]
    )

    kwargs = {
        "vocabulary_size": transformers_config["vocab_size"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "layer_norm_epsilon": transformers_config.get("rms_norm_eps", 1e-6),
        "rope_max_wavelength": transformers_config.get("rope_theta", 1000000),
        "tie_word_embeddings": transformers_config.get(
            "tie_word_embeddings", True
        ),
        "mrope_section": mrope_section,
    }

    # Instantiate vision encoder if config is present.
    if vision_config:
        vision_encoder = Qwen2VLVisionEncoder(
            hidden_size=vision_config.get(
                "hidden_size", transformers_config["hidden_size"]
            ),
            embed_dim=vision_config.get("embed_dim", 1280),
            depth=vision_config.get("depth", 32),
            num_heads=vision_config.get("num_heads", 16),
            patch_size=vision_config.get("spatial_patch_size", 14),
            temporal_patch_size=vision_config.get(
                "temporal_patch_size", 2
            ),
            in_channels=vision_config.get("in_chans", 3),
            mlp_ratio=vision_config.get("mlp_ratio", 4.0),
            spatial_merge_size=vision_config.get(
                "spatial_merge_size", 2
            ),
            name="vision_encoder",
        )
        kwargs["vision_encoder"] = vision_encoder

    return kwargs


def convert_weights(backbone, loader, transformers_config):
    # === Token embedding ===
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
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # === Text decoder layers ===
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention — Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.bias",
            hook_fn=transpose_and_reshape,
        )
        # Attention — Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.bias",
            hook_fn=transpose_and_reshape,
        )
        # Attention — Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.bias",
            hook_fn=transpose_and_reshape,
        )
        # Attention — Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer(
            "sequence_output_layernorm"
        ).scale,
        hf_weight_key="model.norm.weight",
    )

    # === Vision encoder weights (if present) ===
    if backbone.vision_encoder is not None:
        vision = backbone.vision_encoder

        # Patch embedding (Conv3D)
        loader.port_weight(
            keras_variable=vision.patch_embed.proj.kernel,
            hf_weight_key="visual.patch_embed.proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(2, 3, 4, 1, 0)
            ),
        )

        # Vision transformer blocks
        for i in range(vision.depth):
            block = vision.blocks[i]
            prefix = f"visual.blocks.{i}"

            # Layer norms
            loader.port_weight(
                keras_variable=block.norm1.gamma,
                hf_weight_key=f"{prefix}.norm1.weight",
            )
            loader.port_weight(
                keras_variable=block.norm1.beta,
                hf_weight_key=f"{prefix}.norm1.bias",
            )
            loader.port_weight(
                keras_variable=block.norm2.gamma,
                hf_weight_key=f"{prefix}.norm2.weight",
            )
            loader.port_weight(
                keras_variable=block.norm2.beta,
                hf_weight_key=f"{prefix}.norm2.bias",
            )

            # Attention QKV (fused)
            loader.port_weight(
                keras_variable=block.attn.qkv.kernel,
                hf_weight_key=f"{prefix}.attn.qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=block.attn.qkv.bias,
                hf_weight_key=f"{prefix}.attn.qkv.bias",
            )

            # Attention output projection
            loader.port_weight(
                keras_variable=block.attn.proj.kernel,
                hf_weight_key=f"{prefix}.attn.proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=block.attn.proj.bias,
                hf_weight_key=f"{prefix}.attn.proj.bias",
            )

            # MLP
            loader.port_weight(
                keras_variable=block.mlp.fc1.kernel,
                hf_weight_key=f"{prefix}.mlp.fc1.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=block.mlp.fc1.bias,
                hf_weight_key=f"{prefix}.mlp.fc1.bias",
            )
            loader.port_weight(
                keras_variable=block.mlp.fc2.kernel,
                hf_weight_key=f"{prefix}.mlp.fc2.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=block.mlp.fc2.bias,
                hf_weight_key=f"{prefix}.mlp.fc2.bias",
            )

        # Patch merger
        loader.port_weight(
            keras_variable=vision.merger.ln_q.gamma,
            hf_weight_key="visual.merger.ln_q.weight",
        )
        loader.port_weight(
            keras_variable=vision.merger.ln_q.beta,
            hf_weight_key="visual.merger.ln_q.bias",
        )
        loader.port_weight(
            keras_variable=vision.merger.dense1.kernel,
            hf_weight_key="visual.merger.mlp.0.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=vision.merger.dense1.bias,
            hf_weight_key="visual.merger.mlp.0.bias",
        )
        loader.port_weight(
            keras_variable=vision.merger.dense2.kernel,
            hf_weight_key="visual.merger.mlp.2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=vision.merger.dense2.bias,
            hf_weight_key="visual.merger.mlp.2.bias",
        )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

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
