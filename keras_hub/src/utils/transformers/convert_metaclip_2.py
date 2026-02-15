"""Convert MetaCLIP 2 checkpoints from HuggingFace Transformers."""

import numpy as np

from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.models.metaclip_2.metaclip_2_text_encoder import (
    MetaCLIP2TextEncoder,
)
from keras_hub.src.models.metaclip_2.metaclip_2_vision_encoder import (
    MetaCLIP2VisionEncoder,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = MetaCLIP2Backbone


def load_image_converter_config(preset, transformers_config):
    """Load image converter config from HuggingFace preprocessor config."""
    preprocessor_config = load_json(preset, "preprocessor_config.json")
    if preprocessor_config is None:
        return None

    mean = preprocessor_config.get(
        "image_mean", [0.48145466, 0.4578275, 0.40821073]
    )
    std = preprocessor_config.get(
        "image_std", [0.26862954, 0.26130258, 0.27577711]
    )
    rescale_factor = preprocessor_config.get("rescale_factor", 1.0 / 255.0)

    # Calculate scale and offset for normalization
    # The formula is: (pixel * rescale_factor - mean) / std
    # Which can be rewritten as: pixel * (rescale_factor / std) + (-mean / std)
    scale = [rescale_factor / s for s in std]
    offset = [-m / s for m, s in zip(mean, std)]

    # Get image size from vision config or preprocessor config
    if "vision_config" in transformers_config:
        image_size = transformers_config["vision_config"].get("image_size", 224)
    else:
        crop_size = preprocessor_config.get("crop_size", {})
        image_size = crop_size.get("height", 224)

    return {
        "image_size": (image_size, image_size),
        "scale": scale,
        "offset": offset,
        "interpolation": "bicubic",
    }


def convert_backbone_config(transformers_config):
    """Convert HuggingFace config to Keras config."""
    vision_config = transformers_config["vision_config"]
    text_config = transformers_config["text_config"]

    # Get projection_dim from top level or from vision/text config
    projection_dim = transformers_config.get("projection_dim", None)
    if projection_dim is None:
        projection_dim = vision_config.get(
            "projection_dim", text_config.get("projection_dim")
        )

    image_size = vision_config["image_size"]

    return {
        "vision_encoder": MetaCLIP2VisionEncoder(
            patch_size=vision_config["patch_size"],
            hidden_dim=vision_config["hidden_size"],
            num_layers=vision_config["num_hidden_layers"],
            num_heads=vision_config["num_attention_heads"],
            intermediate_dim=vision_config["intermediate_size"],
            intermediate_activation=vision_config.get(
                "hidden_act", "quick_gelu"
            ),
            image_shape=(image_size, image_size, 3),
        ),
        "text_encoder": MetaCLIP2TextEncoder(
            vocabulary_size=text_config["vocab_size"],
            embedding_dim=text_config["hidden_size"],
            hidden_dim=text_config["hidden_size"],
            num_layers=text_config["num_hidden_layers"],
            num_heads=text_config["num_attention_heads"],
            intermediate_dim=text_config["intermediate_size"],
            intermediate_activation=text_config.get("hidden_act", "quick_gelu"),
            max_sequence_length=text_config["max_position_embeddings"],
        ),
        "projection_dim": projection_dim,
    }


def convert_weights(backbone, loader, transformers_config):
    """Convert weights from HuggingFace to Keras format."""

    def port_ln(keras_variable, weight_key):
        loader.port_weight(keras_variable.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        loader.port_weight(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            loader.port_weight(keras_variable.bias, f"{weight_key}.bias")

    def port_mha(keras_variable, weight_key, num_heads, hidden_dim):
        # query
        loader.port_weight(
            keras_variable.query_dense.kernel,
            f"{weight_key}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.query_dense.bias,
            f"{weight_key}.q_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        loader.port_weight(
            keras_variable.key_dense.kernel,
            f"{weight_key}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.key_dense.bias,
            f"{weight_key}.k_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        loader.port_weight(
            keras_variable.value_dense.kernel,
            f"{weight_key}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.value_dense.bias,
            f"{weight_key}.v_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        loader.port_weight(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        loader.port_weight(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    # === Vision Encoder ===
    # Patch embedding (Conv2D kernel needs transpose)
    loader.port_weight(
        backbone.vision_encoder.embedding.patch_embedding.kernel,
        "vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    # Position embedding
    loader.port_weight(
        backbone.vision_encoder.embedding.position_embedding.embeddings,
        "vision_model.embeddings.position_embedding.weight",
    )
    # Class embedding
    loader.port_weight(
        backbone.vision_encoder.embedding.class_embedding,
        "vision_model.embeddings.class_embedding",
    )
    port_ln(
        backbone.vision_encoder.pre_layer_norm,
        "vision_model.pre_layrnorm",
    )
    # Encoder layers
    encoder_layers = backbone.vision_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "vision_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].attention,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm_1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm_2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].dense_1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].dense_2, f"{prefix}.{i}.mlp.fc2")
    # Post layer norm
    port_ln(backbone.vision_post_layer_norm, "vision_model.post_layernorm")
    # Vision projection
    port_dense(backbone.vision_projection, "visual_projection")

    # === Text Encoder ===
    # Token embedding
    loader.port_weight(
        backbone.text_encoder.embedding.token_embedding._embeddings,
        "text_model.embeddings.token_embedding.weight",
    )
    # Position embedding
    loader.port_weight(
        backbone.text_encoder.embedding.position_embedding.position_embeddings,
        "text_model.embeddings.position_embedding.weight",
    )
    # Encoder layers
    encoder_layers = backbone.text_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "text_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].attention,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm_1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm_2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].dense_1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].dense_2, f"{prefix}.{i}.mlp.fc2")
    # Final layer norm
    port_ln(backbone.text_encoder.layer_norm, "text_model.final_layer_norm")
    # Text projection
    port_dense(backbone.text_projection, "text_projection")

    # === Logit Scale ===
    loader.port_weight(backbone.metaclip_2_head.logit_scale, "logit_scale")


def convert_tokenizer(cls, preset, **kwargs):
    """Convert XLM-V tokenizer to Keras Hub MetaCLIP2Tokenizer.

    MetaCLIP 2 uses the XLM-V tokenizer (facebook/xlm-v-base) which is a
    SentencePiece-based multilingual tokenizer.
    """
    # Get the SentencePiece model file
    spm_path = get_file(preset, "sentencepiece.bpe.model")

    # Read the proto
    with open(spm_path, "rb") as f:
        proto = f.read()

    return cls(proto=proto, **kwargs)
