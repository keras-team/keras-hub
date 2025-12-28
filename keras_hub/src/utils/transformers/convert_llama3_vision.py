"""Weight conversion for Llama 3.2 Vision from HuggingFace Transformers.

This module provides utilities to convert Llama 3.2 Vision weights from
the HuggingFace Transformers format to the Keras Hub format.

The conversion handles:
- Vision encoder (ViT) weights
- Vision projector (MLP) weights
- Cross-attention layer weights
- Text backbone weights (Llama3)
"""

import numpy as np

from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Llama3VisionBackbone


def convert_backbone_config(transformers_config):
    """Convert HuggingFace config to Keras Hub config.

    Args:
        transformers_config: Dict from HuggingFace config.json.

    Returns:
        Dict that can be passed to Llama3VisionConfig.
    """
    # Extract vision config
    vision_config = transformers_config.get("vision_config", {})
    text_config = transformers_config.get("text_config", {})

    # Vision encoder configuration
    vision_encoder_config = {
        "hidden_dim": vision_config.get("hidden_size", 1280),
        "num_layers": vision_config.get("num_hidden_layers", 32),
        "num_heads": vision_config.get("num_attention_heads", 16),
        "intermediate_dim": vision_config.get("intermediate_size", 5120),
        "patch_size": vision_config.get("patch_size", 14),
        "image_size": vision_config.get("image_size", 560),
        "num_channels": vision_config.get("num_channels", 3),
        "layer_norm_epsilon": vision_config.get("layer_norm_eps", 1e-6),
    }

    # Text backbone configuration
    text_backbone_config = {
        "vocabulary_size": text_config.get("vocab_size", 128256),
        "num_layers": text_config.get("num_hidden_layers", 40),
        "num_query_heads": text_config.get("num_attention_heads", 32),
        "hidden_dim": text_config.get("hidden_size", 4096),
        "intermediate_dim": text_config.get("intermediate_size", 14336),
        "num_key_value_heads": text_config.get("num_key_value_heads", 8),
        "rope_max_wavelength": text_config.get("rope_theta", 500000),
        "layer_norm_epsilon": text_config.get("rms_norm_eps", 1e-5),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
    }

    # Handle RoPE scaling for Llama 3.1+
    rope_scaling = text_config.get("rope_scaling", None)
    if rope_scaling is not None:
        text_backbone_config["rope_frequency_adjustment_factor"] = (
            rope_scaling.get("factor", 1.0)
        )
        text_backbone_config["rope_low_freq_factor"] = rope_scaling.get(
            "low_freq_factor", 1.0
        )
        text_backbone_config["rope_high_freq_factor"] = rope_scaling.get(
            "high_freq_factor", 4.0
        )
        text_backbone_config["rope_pretraining_sequence_length"] = (
            rope_scaling.get("original_max_position_embeddings", 8192)
        )

    # Cross-attention layers (every 5th layer starting from layer 3)
    # This varies depending on the model size
    num_text_layers = text_config.get("num_hidden_layers", 40)
    cross_attention_layers = transformers_config.get(
        "cross_attention_layers",
        [i for i in range(3, num_text_layers, 5)],  # Default: every 5th layer
    )

    return {
        "vision_encoder_config": vision_encoder_config,
        "text_config": text_backbone_config,
        "cross_attention_layers": cross_attention_layers,
    }


def convert_weights(backbone, loader, transformers_config):
    """Convert HuggingFace weights to Keras Hub weights.

    Args:
        backbone: Llama3VisionBackbone instance.
        loader: SafeTensorLoader for loading weights.
        transformers_config: HuggingFace config dict.

    Returns:
        The backbone with loaded weights.
    """
    _convert_vision_encoder_weights(backbone, loader)
    _convert_vision_projector_weights(backbone, loader)
    _convert_cross_attention_weights(backbone, loader)
    _convert_text_backbone_weights(backbone, loader)

    return backbone


def _convert_vision_encoder_weights(backbone, loader):
    """Convert vision encoder (ViT) weights."""
    encoder = backbone.vision_encoder

    # Patch embedding (Conv2D)
    loader.port_weight(
        keras_variable=encoder.patch_embedding.kernel,
        hf_weight_key="vision_model.patch_embedding.weight",
        # HF: (out_channels, in_channels, H, W)
        # -> Keras: (H, W, in_channels, out_channels)
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    # Position embedding
    loader.port_weight(
        keras_variable=encoder.position_embedding.embeddings,
        hf_weight_key="vision_model.position_embedding.weight",
    )

    # Transformer layers
    if encoder.is_two_stage:
        # Two-stage encoder: local + global
        _convert_vision_transformer_layers(
            encoder.local_transformer_layers,
            loader,
            prefix="vision_model.encoder.layers",
            start_idx=0,
        )
        _convert_vision_transformer_layers(
            encoder.global_transformer_layers,
            loader,
            prefix="vision_model.global_encoder.layers",
            start_idx=0,
        )
    else:
        # Single-stage encoder
        _convert_vision_transformer_layers(
            encoder.transformer_layers,
            loader,
            prefix="vision_model.encoder.layers",
            start_idx=0,
        )

    # Final layer norm
    loader.port_weight(
        keras_variable=encoder.layer_norm.gamma,
        hf_weight_key="vision_model.post_layernorm.weight",
    )
    loader.port_weight(
        keras_variable=encoder.layer_norm.beta,
        hf_weight_key="vision_model.post_layernorm.bias",
    )


def _convert_vision_transformer_layers(layers, loader, prefix, start_idx):
    """Convert vision transformer layer weights."""

    def transpose(x, _):
        return np.transpose(x)

    for i, layer in enumerate(layers):
        idx = start_idx + i
        layer_prefix = f"{prefix}.{idx}"

        # Pre-norm (layer_norm_1)
        loader.port_weight(
            keras_variable=layer._self_attention_layernorm.gamma,
            hf_weight_key=f"{layer_prefix}.layer_norm1.weight",
        )
        loader.port_weight(
            keras_variable=layer._self_attention_layernorm.beta,
            hf_weight_key=f"{layer_prefix}.layer_norm1.bias",
        )

        # Self-attention
        loader.port_weight(
            keras_variable=layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"{layer_prefix}.self_attn.q_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"{layer_prefix}.self_attn.k_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"{layer_prefix}.self_attn.v_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"{layer_prefix}.self_attn.out_proj.weight",
            hook_fn=transpose,
        )

        # FFN norm (layer_norm_2)
        loader.port_weight(
            keras_variable=layer._feedforward_layernorm.gamma,
            hf_weight_key=f"{layer_prefix}.layer_norm2.weight",
        )
        loader.port_weight(
            keras_variable=layer._feedforward_layernorm.beta,
            hf_weight_key=f"{layer_prefix}.layer_norm2.bias",
        )

        # FFN
        loader.port_weight(
            keras_variable=layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{layer_prefix}.mlp.fc1.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{layer_prefix}.mlp.fc1.bias",
        )
        loader.port_weight(
            keras_variable=layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{layer_prefix}.mlp.fc2.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=layer._feedforward_output_dense.bias,
            hf_weight_key=f"{layer_prefix}.mlp.fc2.bias",
        )


def _convert_vision_projector_weights(backbone, loader):
    """Convert vision projector (multi-modal projector) weights."""
    projector = backbone.vision_projector

    def transpose(x, _):
        return np.transpose(x)

    # Dense 1 (input projection)
    loader.port_weight(
        keras_variable=projector.dense_1.kernel,
        hf_weight_key="multi_modal_projector.linear_1.weight",
        hook_fn=transpose,
    )
    loader.port_weight(
        keras_variable=projector.dense_1.bias,
        hf_weight_key="multi_modal_projector.linear_1.bias",
    )

    # Dense 2 (output projection)
    loader.port_weight(
        keras_variable=projector.dense_2.kernel,
        hf_weight_key="multi_modal_projector.linear_2.weight",
        hook_fn=transpose,
    )
    loader.port_weight(
        keras_variable=projector.dense_2.bias,
        hf_weight_key="multi_modal_projector.linear_2.bias",
    )


def _convert_cross_attention_weights(backbone, loader):
    """Convert cross-attention layer weights."""

    def transpose(x, _):
        return np.transpose(x)

    for layer_idx, cross_attn in backbone.cross_attention_blocks.items():
        prefix = f"language_model.model.layers.{layer_idx}.cross_attn"

        # Query norm
        loader.port_weight(
            keras_variable=cross_attn.query_norm.scale,
            hf_weight_key=f"{prefix}.q_norm.weight",
        )

        # KV norm
        loader.port_weight(
            keras_variable=cross_attn.kv_norm.scale,
            hf_weight_key=f"{prefix}.kv_norm.weight",
        )

        # Query projection
        loader.port_weight(
            keras_variable=cross_attn.query_dense.kernel,
            hf_weight_key=f"{prefix}.q_proj.weight",
            hook_fn=transpose,
        )

        # Key projection
        loader.port_weight(
            keras_variable=cross_attn.key_dense.kernel,
            hf_weight_key=f"{prefix}.k_proj.weight",
            hook_fn=transpose,
        )

        # Value projection
        loader.port_weight(
            keras_variable=cross_attn.value_dense.kernel,
            hf_weight_key=f"{prefix}.v_proj.weight",
            hook_fn=transpose,
        )

        # Output projection
        loader.port_weight(
            keras_variable=cross_attn.output_dense.kernel,
            hf_weight_key=f"{prefix}.o_proj.weight",
            hook_fn=transpose,
        )

        # Gate parameter
        loader.port_weight(
            keras_variable=cross_attn.gate,
            hf_weight_key=f"{prefix}.gate",
        )


def _convert_text_backbone_weights(backbone, loader):
    """Convert text backbone (Llama3) weights."""
    text_backbone = backbone.text_backbone

    # Token embedding
    loader.port_weight(
        keras_variable=text_backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="language_model.model.embed_tokens.weight",
    )

    # Optional: reverse embeddings for untied LM head
    if not text_backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=text_backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="language_model.lm_head.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Transformer layers
    for i in range(text_backbone.num_layers):
        decoder_layer = text_backbone.get_layer(f"transformer_layer_{i}")
        prefix = f"language_model.model.layers.{i}"

        # Self-attention layer norm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"{prefix}.input_layernorm.weight",
        )

        # FFN layer norm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"{prefix}.post_attention_layernorm.weight",
        )

        # Self-attention projections
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # FFN
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"{prefix}.mlp.gate_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{prefix}.mlp.up_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{prefix}.mlp.down_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

    # Final layer norm
    loader.port_weight(
        keras_variable=text_backbone.get_layer(
            "sequence_output_layernorm"
        ).scale,
        hf_weight_key="language_model.model.norm.weight",
    )


def convert_tokenizer(cls, preset, **kwargs):
    """Convert tokenizer from HuggingFace format.

    Args:
        cls: Tokenizer class to instantiate.
        preset: Path to the preset directory.
        **kwargs: Additional arguments for the tokenizer.

    Returns:
        Instantiated tokenizer.
    """
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    # Handle different merge formats
    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(merge) for merge in merges]

    # Load special tokens
    special_tokens = set()
    for token in tokenizer_config["added_tokens"]:
        if not token["content"].startswith("<|reserved_special_token_"):
            vocab[token["content"]] = token["id"]
            special_tokens.add(token["content"])

    # Load start/end tokens from config
    tokenizer_config2 = load_json(preset, "tokenizer_config.json")
    bos_token = tokenizer_config2["bos_token"]
    eos_token = tokenizer_config2["eos_token"]

    kwargs.update(
        {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "misc_special_tokens": special_tokens,
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)


def load_image_converter_config(preset, transformers_config):
    """Load image converter configuration.

    Args:
        preset: Path to preset directory.
        transformers_config: HuggingFace config dict.

    Returns:
        Dict with image converter settings or None.
    """
    if "vision_config" not in transformers_config:
        return None

    try:
        preprocessor_config = load_json(preset, "preprocessor_config.json")
    except (FileNotFoundError, ValueError):
        # Fallback to default values when file doesn't exist
        # or preset is invalid
        return {
            "image_size": transformers_config["vision_config"].get(
                "image_size", 560
            ),
            "scale": 1.0 / 255.0,
            "offset": 0.0,
        }

    image_size = transformers_config["vision_config"].get("image_size", 560)
    mean = preprocessor_config.get("image_mean", [0.5, 0.5, 0.5])
    std = preprocessor_config.get("image_std", [0.5, 0.5, 0.5])
    rescale_factor = preprocessor_config.get("rescale_factor", 1.0 / 255.0)

    # Calculate offset and scale for normalization
    # normalize = (pixel * rescale_factor - mean) / std
    # rewrite as: pixel * (rescale_factor / std) + (-mean / std)
    offset = [(-m / s) for m, s in zip(mean, std)]
    scale = [(rescale_factor / s) for s in std]

    return {
        "image_size": image_size,
        "scale": scale,
        "offset": offset,
    }
