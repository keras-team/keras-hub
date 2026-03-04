import numpy as np

from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = T5Gemma2Backbone


def load_image_converter_config(preset, transformers_config):
    """Load image converter config from HF preprocessor_config.json."""
    encoder_config = transformers_config.get("encoder", {})
    if "vision_config" not in encoder_config:
        return None
    preprocessor_config = load_json(preset, "preprocessor_config.json")
    mean = preprocessor_config["image_mean"]
    std = preprocessor_config["image_std"]
    rescale_factor = preprocessor_config["rescale_factor"]
    offset = [(-m / s) for m, s in zip(mean, std)]
    scale = [(s * rescale_factor) for s in std]
    image_size = encoder_config["vision_config"].get("image_size", 896)
    return {
        "image_size": (image_size, image_size),
        "scale": scale,
        "offset": offset,
    }


def convert_backbone_config(transformers_config):
    """Convert a HuggingFace T5Gemma2 config to KerasHub backbone config."""
    # T5Gemma2EncoderConfig is Gemma3Config with text params at
    # encoder["text_config"]; decoder is Gemma3TextConfig (flat).
    encoder_config = transformers_config["encoder"]
    enc_text = encoder_config["text_config"]
    decoder_config = transformers_config["decoder"]

    hidden_activation = decoder_config.get(
        "hidden_activation", "gelu_pytorch_tanh"
    )
    if hidden_activation == "gelu_pytorch_tanh":
        hidden_activation = "gelu_approximate"

    # Vision encoder (optional).
    vision_encoder = None
    if "vision_config" in encoder_config:
        from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
            Gemma3VisionEncoder,
        )

        vision_config = encoder_config["vision_config"]
        vision_encoder = Gemma3VisionEncoder(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            num_heads=vision_config["num_attention_heads"],
            hidden_dim=vision_config["hidden_size"],
            num_layers=vision_config["num_hidden_layers"],
            intermediate_dim=vision_config["intermediate_size"],
            output_dim=enc_text["hidden_size"],
            pool_size=int(
                vision_config["image_size"]
                // vision_config["patch_size"]
                // int(encoder_config.get("mm_tokens_per_image", 256) ** 0.5)
            ),
            layer_norm_epsilon=vision_config.get("layer_norm_eps", 1e-6),
        )

    backbone_config = {
        "vocabulary_size": decoder_config["vocab_size"],
        "encoder_hidden_dim": enc_text["hidden_size"],
        "encoder_intermediate_dim": enc_text["intermediate_size"],
        "encoder_num_layers": enc_text["num_hidden_layers"],
        "encoder_num_attention_heads": enc_text["num_attention_heads"],
        "encoder_num_key_value_heads": enc_text["num_key_value_heads"],
        "encoder_head_dim": enc_text["head_dim"],
        "encoder_layer_types": enc_text["layer_types"],
        "decoder_hidden_dim": decoder_config["hidden_size"],
        "decoder_intermediate_dim": decoder_config["intermediate_size"],
        "decoder_num_layers": decoder_config["num_hidden_layers"],
        "decoder_num_attention_heads": decoder_config["num_attention_heads"],
        "decoder_num_key_value_heads": decoder_config["num_key_value_heads"],
        "decoder_head_dim": decoder_config["head_dim"],
        "decoder_layer_types": decoder_config["layer_types"],
        "dropout_rate": decoder_config["dropout_rate"],
        "rms_norm_eps": decoder_config["rms_norm_eps"],
        "query_pre_attn_scalar": decoder_config["query_pre_attn_scalar"],
        "tie_word_embeddings": transformers_config.get(
            "tie_word_embeddings", True
        ),
        "attention_bias": decoder_config["attention_bias"],
        "hidden_activation": hidden_activation,
        "initializer_range": decoder_config["initializer_range"],
        "attention_dropout": decoder_config["attention_dropout"],
        "sliding_window": decoder_config["sliding_window"],
        "cross_attention_hidden_size": enc_text["hidden_size"],
        "attn_logit_softcapping": decoder_config["attn_logit_softcapping"],
        "final_logit_softcapping": decoder_config["final_logit_softcapping"],
        "rope_max_wavelength": decoder_config.get("rope_theta", 10000.0),
        "global_rope_scaling_factor": decoder_config.get("rope_parameters", {})
        .get("full_attention", {})
        .get("factor", 1.0),
        "encoder_rope_max_wavelength": enc_text.get("rope_parameters", {})
        .get("sliding_attention", {})
        .get("rope_theta", None),
        "encoder_global_rope_scaling_factor": enc_text.get(
            "rope_parameters", {}
        )
        .get("full_attention", {})
        .get("factor", None),
        "use_query_key_norm": True,
        "vision_encoder": vision_encoder,
        "eoi_token_index": transformers_config.get("eoi_token_index", 256000),
    }
    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """Convert T5Gemma2 weights from HuggingFace to KerasHub."""

    def transpose(x, shape):
        return np.transpose(x)

    # === Vision encoder weights ===
    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        image_encoder = vision_encoder.get_layer("image_encoder")

        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.kernel,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.patch_embedding.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.bias,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.patch_embedding.bias",
        )
        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.position_embedding.embeddings,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.position_embedding.weight",
        )

        for i in range(image_encoder.num_layers):
            hf_vit = f"encoder.vision_tower.vision_model.encoder.layers.{i}"
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.gamma,
                hf_weight_key=f"{hf_vit}.layer_norm1.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.beta,
                hf_weight_key=f"{hf_vit}.layer_norm1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.query_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.q_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.query_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.q_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.k_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.k_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.value_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.v_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.value_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.v_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.out_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.out_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.gamma,
                hf_weight_key=f"{hf_vit}.layer_norm2.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.beta,
                hf_weight_key=f"{hf_vit}.layer_norm2.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.kernel,
                hf_weight_key=f"{hf_vit}.mlp.fc1.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.bias,
                hf_weight_key=f"{hf_vit}.mlp.fc1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.kernel,
                hf_weight_key=f"{hf_vit}.mlp.fc2.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.bias,
                hf_weight_key=f"{hf_vit}.mlp.fc2.bias",
            )

        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.gamma,
            hf_weight_key="encoder.vision_tower.vision_model.post_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.beta,
            hf_weight_key="encoder.vision_tower.vision_model.post_layernorm.bias",
        )

        # Multi-modal projector.
        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_soft_embedding_norm.scale,
            hf_weight_key="encoder.multi_modal_projector.mm_soft_emb_norm.weight",
        )
        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_input_projection.kernel,
            hf_weight_key="encoder.multi_modal_projector.mm_input_projection_weight",
        )

        # EOI embeddings.
        loader.port_weight(
            keras_variable=backbone.encoder_eoi_embedding,
            hf_weight_key="encoder.text_model.embed_tokens.eoi_embedding",
        )
        loader.port_weight(
            keras_variable=backbone.decoder_eoi_embedding,
            hf_weight_key="decoder.embed_tokens.eoi_embedding",
        )

    # === Text encoder weights ===
    # Token embeddings.
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="encoder.text_model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.decoder_token_embedding.embeddings,
        hf_weight_key="decoder.embed_tokens.weight",
    )

    # Encoder (weights under encoder.text_model.*).
    loader.port_weight(
        keras_variable=backbone.encoder_norm.scale,
        hf_weight_key="encoder.text_model.norm.weight",
    )
    for i in range(backbone.encoder_num_layers):
        layer = backbone.get_layer(f"encoder_layer_{i}")
        hf_prefix = f"encoder.text_model.layers.{i}"

        # Self-attention Q/K/V/O projections.
        loader.port_weight(
            keras_variable=layer.self_attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.o_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )

        # Q/K normalization (Gemma3-style).
        loader.port_weight(
            keras_variable=layer.self_attn.query_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer.self_attn.key_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.k_norm.weight",
        )

        # MLP.
        loader.port_weight(
            keras_variable=layer.mlp.gate_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.gate_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.up_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.up_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.down_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.down_proj.weight",
            hook_fn=lambda w, s: w.T,
        )

        # Layer norms.
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_feedforward_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_feedforward_layernorm.weight"),
        )

    # Decoder (weights directly under decoder.*).
    loader.port_weight(
        keras_variable=backbone.decoder_norm.scale,
        hf_weight_key="decoder.norm.weight",
    )
    for i in range(backbone.decoder_num_layers):
        layer = backbone.get_layer(f"decoder_layer_{i}")
        hf_prefix = f"decoder.layers.{i}"

        # Merged attention (self+cross uses a single self_attn layer).
        loader.port_weight(
            keras_variable=layer.merged_attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.o_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )

        # Q/K normalization.
        loader.port_weight(
            keras_variable=layer.merged_attn.query_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.key_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.k_norm.weight",
        )

        # MLP.
        loader.port_weight(
            keras_variable=layer.mlp.gate_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.gate_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.up_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.up_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.down_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.down_proj.weight",
            hook_fn=lambda w, s: w.T,
        )

        # Layer norms (no cross-attn norms — merged into self_attn).
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_feedforward_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_feedforward_layernorm.weight"),
        )


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a T5Gemma2 tokenizer."""
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
