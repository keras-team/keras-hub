import numpy as np

from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_vision_encoder import CLIPVisionEncoder
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = CLIPBackbone


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
    image_size = transformers_config["vision_config"].get("image_size", 224)

    # (pixel * rescale_factor - mean) / std
    #   == pixel * (rescale_factor / std) + (-mean / std)
    return {
        "image_size": (image_size, image_size),
        "scale": [rescale_factor / s for s in std],
        "offset": [-m / s for m, s in zip(mean, std)],
        "interpolation": "bicubic",
    }


def convert_backbone_config(transformers_config):
    """Convert HuggingFace config to Keras config."""
    vision_config = transformers_config["vision_config"]
    text_config = transformers_config["text_config"]

    projection_dim = transformers_config.get("projection_dim")
    if projection_dim is None:
        projection_dim = vision_config.get(
            "projection_dim", text_config.get("projection_dim")
        )

    image_size = vision_config["image_size"]
    return {
        "vision_encoder": CLIPVisionEncoder(
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
        "text_encoder": CLIPTextEncoder(
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
        head_dim = hidden_dim // num_heads
        for keras_name, hf_name in (
            ("query_dense", "q_proj"),
            ("key_dense", "k_proj"),
            ("value_dense", "v_proj"),
        ):
            dense = getattr(keras_variable, keras_name)
            loader.port_weight(
                dense.kernel,
                f"{weight_key}.{hf_name}.weight",
                hook_fn=lambda x, _: np.reshape(
                    x.T, (hidden_dim, num_heads, head_dim)
                ),
            )
            loader.port_weight(
                dense.bias,
                f"{weight_key}.{hf_name}.bias",
                hook_fn=lambda x, _: np.reshape(x, (num_heads, head_dim)),
            )
        loader.port_weight(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, head_dim, hidden_dim)
            ),
        )
        loader.port_weight(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    def port_encoder_layers(encoder_layers, prefix):
        for i, layer in enumerate(encoder_layers):
            port_mha(
                layer.attention,
                f"{prefix}.{i}.self_attn",
                layer.num_heads,
                layer.hidden_dim,
            )
            port_ln(layer.layer_norm_1, f"{prefix}.{i}.layer_norm1")
            port_ln(layer.layer_norm_2, f"{prefix}.{i}.layer_norm2")
            port_dense(layer.dense_1, f"{prefix}.{i}.mlp.fc1")
            port_dense(layer.dense_2, f"{prefix}.{i}.mlp.fc2")

    # === Vision Encoder ===
    vision_encoder = backbone.vision_encoder
    embedding = vision_encoder.embedding
    loader.port_weight(
        embedding.patch_embedding.kernel,
        "vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        embedding.position_embedding.embeddings,
        "vision_model.embeddings.position_embedding.weight",
    )
    loader.port_weight(
        embedding.class_embedding,
        "vision_model.embeddings.class_embedding",
    )
    # `position_ids` is a non-persistent buffer in HF and is often absent
    # from safetensors checkpoints, so we set it directly.
    embedding.position_ids.assign(
        np.arange(embedding.num_positions)[np.newaxis, :]
    )
    port_ln(vision_encoder.pre_layer_norm, "vision_model.pre_layrnorm")
    port_encoder_layers(
        vision_encoder.encoder_layers, "vision_model.encoder.layers"
    )
    port_ln(vision_encoder.layer_norm, "vision_model.post_layernorm")
    port_dense(backbone.vision_projection, "visual_projection")

    # === Text Encoder ===
    text_encoder = backbone.text_encoder
    loader.port_weight(
        text_encoder.embedding.token_embedding._embeddings,
        "text_model.embeddings.token_embedding.weight",
    )
    loader.port_weight(
        text_encoder.embedding.position_embedding.position_embeddings,
        "text_model.embeddings.position_embedding.weight",
    )
    port_encoder_layers(
        text_encoder.encoder_layers, "text_model.encoder.layers"
    )
    port_ln(text_encoder.layer_norm, "text_model.final_layer_norm")
    port_dense(backbone.text_projection, "text_projection")

    # === Logit Scale ===
    loader.port_weight(backbone.clip_head.logit_scale, "logit_scale")


def convert_tokenizer(cls, preset, **kwargs):
    """Convert HuggingFace CLIP BPE tokenizer to KerasHub `CLIPTokenizer`."""
    return cls(
        vocabulary=get_file(preset, "vocab.json"),
        merges=get_file(preset, "merges.txt"),
        **{"pad_with_end_token": True, **kwargs},
    )
