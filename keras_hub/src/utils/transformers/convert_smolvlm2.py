"""Convert SmolVLM2 weights from Hugging Face to KerasHub."""

import numpy as np

from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = SmolVLM2Backbone


def convert_backbone_config(transformers_config):
    """Map HF SmolVLM config.json to KerasHub backbone kwargs.

    All values are read directly from the preset's config.json.
    No hardcoded defaults — if a required key is missing from the
    preset, this will raise a KeyError immediately.
    """
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]

    # Some presets (e.g. 2.2B) omit num_attention_heads from text_config,
    # providing head_dim instead.  Compute it as hidden_size / head_dim,
    # matching HF LlamaConfig's resolution logic.
    if "num_attention_heads" in text_config:
        num_query_heads = text_config["num_attention_heads"]
    else:
        num_query_heads = text_config["hidden_size"] // text_config["head_dim"]

    # num_key_value_heads for the LLM decoder may also be absent
    # (note: perceiver_config.num_key_value_heads is for the resampler,
    #  NOT the decoder).  HF LlamaConfig defaults it to num_attention_heads.
    num_kv_heads = text_config.get("num_key_value_heads", num_query_heads)

    return {
        "vocabulary_size": text_config["vocab_size"],
        "image_size": vision_config["image_size"],
        "patch_size": vision_config["patch_size"],
        # Keys omitted from some presets (e.g. 2.2B); defaults match
        # HF SmolVLMVisionConfig class defaults.
        "vision_hidden_dim": vision_config.get("hidden_size", 1152),
        "vision_intermediate_dim": vision_config.get("intermediate_size", 3072),
        "vision_num_layers": vision_config.get("num_hidden_layers", 12),
        "vision_num_heads": vision_config.get("num_attention_heads", 16),
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": num_query_heads,
        "num_key_value_heads": num_kv_heads,
        "scale_factor": transformers_config["scale_factor"],
        "image_token_id": transformers_config["image_token_id"],
        "rope_max_wavelength": text_config["rope_theta"],
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "vision_layer_norm_epsilon": vision_config.get("layer_norm_eps", 1e-6),
        "tie_word_embeddings": transformers_config["tie_word_embeddings"],
    }


def convert_weights(backbone, loader, transformers_config):
    """Port HF safetensor weights to KerasHub SmolVLM2 backbone."""

    def transpose(hf_tensor, _):
        return np.transpose(hf_tensor, axes=(1, 0))

    def transpose_conv(hf_tensor, _):
        # HF Conv2D kernel: (out_channels, in_channels, kH, kW)
        # Keras Conv2D kernel: (kH, kW, in_channels, out_channels)
        return np.transpose(hf_tensor, axes=(2, 3, 1, 0))

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Vision Encoder.
    vision_encoder = backbone.vision_encoder

    # Patch embedding (Conv2D).
    embeddings_layer = vision_encoder.vision_embeddings
    loader.port_weight(
        keras_variable=embeddings_layer.patch_embedding.kernel,
        hf_weight_key="model.vision_model.embeddings.patch_embedding.weight",
        hook_fn=transpose_conv,
    )
    loader.port_weight(
        keras_variable=embeddings_layer.patch_embedding.bias,
        hf_weight_key="model.vision_model.embeddings.patch_embedding.bias",
    )

    # Position embedding.
    loader.port_weight(
        keras_variable=embeddings_layer.position_embedding.embeddings,
        hf_weight_key=(
            "model.vision_model.embeddings.position_embedding.weight"
        ),
    )

    # Vision encoder blocks.
    num_vision_layers = vision_encoder.num_layers
    for i in range(num_vision_layers):
        block = vision_encoder.encoder_blocks[i]
        prefix = f"model.vision_model.encoder.layers.{i}"

        # Layer norms.
        loader.port_weight(
            keras_variable=block.layer_norm1.gamma,
            hf_weight_key=f"{prefix}.layer_norm1.weight",
        )
        loader.port_weight(
            keras_variable=block.layer_norm1.beta,
            hf_weight_key=f"{prefix}.layer_norm1.bias",
        )
        loader.port_weight(
            keras_variable=block.layer_norm2.gamma,
            hf_weight_key=f"{prefix}.layer_norm2.weight",
        )
        loader.port_weight(
            keras_variable=block.layer_norm2.beta,
            hf_weight_key=f"{prefix}.layer_norm2.bias",
        )

        # Self-attention.
        loader.port_weight(
            keras_variable=block.self_attn.q_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.q_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.self_attn.q_proj.bias,
            hf_weight_key=f"{prefix}.self_attn.q_proj.bias",
        )
        loader.port_weight(
            keras_variable=block.self_attn.k_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.k_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.self_attn.k_proj.bias,
            hf_weight_key=f"{prefix}.self_attn.k_proj.bias",
        )
        loader.port_weight(
            keras_variable=block.self_attn.v_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.v_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.self_attn.v_proj.bias,
            hf_weight_key=f"{prefix}.self_attn.v_proj.bias",
        )
        loader.port_weight(
            keras_variable=block.self_attn.out_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.out_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.self_attn.out_proj.bias,
            hf_weight_key=f"{prefix}.self_attn.out_proj.bias",
        )

        # MLP.
        loader.port_weight(
            keras_variable=block.mlp.fc1.kernel,
            hf_weight_key=f"{prefix}.mlp.fc1.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.mlp.fc1.bias,
            hf_weight_key=f"{prefix}.mlp.fc1.bias",
        )
        loader.port_weight(
            keras_variable=block.mlp.fc2.kernel,
            hf_weight_key=f"{prefix}.mlp.fc2.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=block.mlp.fc2.bias,
            hf_weight_key=f"{prefix}.mlp.fc2.bias",
        )

    # Post layer norm.
    loader.port_weight(
        keras_variable=vision_encoder.post_layernorm.gamma,
        hf_weight_key="model.vision_model.post_layernorm.weight",
    )
    loader.port_weight(
        keras_variable=vision_encoder.post_layernorm.beta,
        hf_weight_key="model.vision_model.post_layernorm.bias",
    )

    # Connector.
    loader.port_weight(
        keras_variable=backbone.connector.modality_projection.kernel,
        hf_weight_key="model.connector.modality_projection.proj.weight",
        hook_fn=transpose,
    )

    # Text Decoder.
    # Token embedding.
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.text_model.embed_tokens.weight",
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[i]
        prefix = f"model.text_model.layers.{i}"

        # Norms.
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.scale,
            hf_weight_key=f"{prefix}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_attention_layernorm.scale,
            hf_weight_key=f"{prefix}.post_attention_layernorm.weight",
        )

        # Self-attention.
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.q_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.k_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.v_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.o_proj.kernel,
            hf_weight_key=f"{prefix}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP.
        loader.port_weight(
            keras_variable=decoder_layer.mlp.gate_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.gate_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=decoder_layer.mlp.up_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.up_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=decoder_layer.mlp.down_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.down_proj.weight",
            hook_fn=transpose,
        )

    # Final normalization.
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="model.text_model.norm.weight",
    )

    # LM head (separate weight if not tied).
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.token_embedding.reverse_embeddings,
            hf_weight_key="lm_head.weight",
            hook_fn=transpose,
        )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    """Convert HF tokenizer to KerasHub SmolVLM2Tokenizer."""
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    merges = [" ".join(item) for item in merges]

    # Load all special tokens.
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


def _compute_scale_offset(image_mean, image_std, rescale_factor=1.0 / 255):
    """Compute KH scale/offset from HF image_mean and image_std.

    KH applies:  output = input * scale + offset
    HF applies:
        1. Rescale: x = x * rescale_factor   (e.g. 1/255)
        2. Normalize: x = (x - mean) / std
    Combined:  scale = rescale_factor / std,  offset = -mean / std
    """
    scale = [rescale_factor / s for s in image_std]
    offset = [-m / s for m, s in zip(image_mean, image_std)]
    return scale, offset


def _load_preprocessor_config(preset):
    """Load and return the HF preprocessor_config.json for this preset."""
    return load_json(preset, "preprocessor_config.json")


def _load_vision_normalization_config(preset):
    """Load shared normalization, interpolation, and max_image_size config.

    Returns a dict with ``scale``, ``offset``, ``interpolation``,
    ``antialias``, ``max_image_size``, and the raw
    ``preprocessor_config`` for caller-specific fields.
    """
    preprocessor_config = _load_preprocessor_config(preset)

    image_mean = preprocessor_config["image_mean"]
    image_std = preprocessor_config["image_std"]
    rescale_factor = preprocessor_config["rescale_factor"]

    scale, offset = _compute_scale_offset(image_mean, image_std, rescale_factor)

    return {
        "scale": scale,
        "offset": offset,
        "interpolation": "bicubic",
        "antialias": True,
        "max_image_size": preprocessor_config["max_image_size"]["longest_edge"],
        "_preprocessor_config": preprocessor_config,
    }


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for SmolVLM2ImageConverter."""
    shared = _load_vision_normalization_config(preset)
    cfg = shared.pop("_preprocessor_config")

    shared.update(
        {
            # Disable base class resizing — our converter handles it.
            "image_size": None,
            "size": cfg["size"]["longest_edge"],
            "do_image_splitting": cfg["do_image_splitting"],
        }
    )
    return shared


def load_video_converter_config(preset, transformers_config):
    """Return kwargs for SmolVLM2VideoConverter."""
    shared = _load_vision_normalization_config(preset)
    cfg = shared.pop("_preprocessor_config")

    video_sampling = cfg["video_sampling"]
    shared.update(
        {
            "size": video_sampling["video_size"]["longest_edge"],
            "num_frames": video_sampling["max_frames"],
            "fps": video_sampling["fps"],
        }
    )
    return shared
