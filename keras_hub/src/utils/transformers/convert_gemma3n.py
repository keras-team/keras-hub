import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.utils.preset_utils import get_file

# Define the Keras backbone class we are targeting.
backbone_cls = Gemma3nBackbone


def convert_backbone_config(transformers_config):
    """
    Converts a Hugging Face Gemma3n config to a Keras Gemma3nBackbone config.
    """
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    audio_config = transformers_config["audio_config"]

    # Map all explicit parameters required by our Keras backbone's constructor.
    backbone_config = {
        # Text Model Parameters
        "vocabulary_size": text_config["vocab_size"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"][
            0
        ],  # Assuming same for all layers for now
        "head_dim": text_config["head_dim"],
        # Per-Layer Input Parameters
        "vocab_size_per_layer_input": text_config["vocab_size_per_layer_input"],
        "hidden_size_per_layer_input": text_config[
            "hidden_size_per_layer_input"
        ],
        # Vision Embedder Parameters
        "vision_vocab_offset": vision_config["vocab_offset"],
        "vision_vocab_size": vision_config["vocab_size"],
        # Audio Embedder Parameters
        "audio_vocab_offset": audio_config["vocab_offset"],
        "audio_vocab_size": audio_config["vocab_size"],
        # Architectural Details from Text Config
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "attention_dropout": text_config["attention_dropout"],
        "attention_bias": text_config["attention_bias"],
        "laurel_rank": text_config["laurel_rank"],
        "altup_num_inputs": text_config["altup_num_inputs"],
        "altup_active_idx": text_config["altup_active_idx"],
        "altup_coef_clip": text_config.get(
            "altup_coef_clip"
        ),  # Use .get for optional keys
        "altup_correct_scale": text_config["altup_correct_scale"],
        "layer_types": text_config["layer_types"],
        "activation_sparsity": text_config["activation_sparsity_pattern"][
            0
        ],  # Assuming same for all layers
    }

    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """
    Ports weights from a Hugging Face checkpoint to the Keras Gemma3nBackbone.

    Args:
        backbone: An instance of the Keras Gemma3nBackbone.
        loader: A utility object for loading weights.
        transformers_config: The Hugging Face model config.
    """

    def transpose(hf_tensor):
        return np.transpose(hf_tensor)

    # --- Main Embeddings ---
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="language_model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.per_layer_embeddings.embeddings,
        hf_weight_key="language_model.embed_tokens_per_layer.weight",
    )

    # --- Multimodal Embedders ---
    if backbone.vision_embedder:
        loader.port_weight(
            keras_variable=backbone.vision_embedder.embedding.embeddings,
            hf_weight_key="embed_vision.embedding.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.hard_embedding_norm.scale,
            hf_weight_key="embed_vision.hard_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.soft_embedding_norm.scale,
            hf_weight_key="embed_vision.soft_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.embedding_projection.kernel,
            hf_weight_key="embed_vision.embedding_projection.weight",
            hook_fn=transpose,
        )

    if backbone.audio_embedder:
        loader.port_weight(
            keras_variable=backbone.audio_embedder.embedding.embeddings,
            hf_weight_key="embed_audio.embedding.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.hard_embedding_norm.scale,
            hf_weight_key="embed_audio.hard_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.soft_embedding_norm.scale,
            hf_weight_key="embed_audio.soft_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.embedding_projection.kernel,
            hf_weight_key="embed_audio.embedding_projection.weight",
            hook_fn=transpose,
        )

    # --- Initial AltUp Projections ---
    for i in range(backbone.altup_num_inputs - 1):
        loader.port_weight(
            keras_variable=backbone.altup_projections[i].kernel,
            hf_weight_key=f"language_model.altup_projections.{i}.weight",
            hook_fn=transpose,
        )

    # --- Transformer Decoder Layers ---
    for i in range(backbone.num_hidden_layers):
        keras_layer = backbone.get_layer(f"gemma3n_transformer_decoder_{i}")
        hf_prefix = f"language_model.layers.{i}"

        # Norms
        loader.port_weight(
            keras_layer.input_layernorm.scale,
            f"{hf_prefix}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_layer.post_attention_layernorm.scale,
            f"{hf_prefix}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            keras_layer.pre_feedforward_layernorm.scale,
            f"{hf_prefix}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            keras_layer.post_feedforward_layernorm.scale,
            f"{hf_prefix}.post_feedforward_layernorm.weight",
        )

        # Attention
        attn = keras_layer.self_attn
        loader.port_weight(
            attn.q_proj.kernel,
            f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            attn.k_proj.kernel,
            f"{hf_prefix}.self_attn.k_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            attn.v_proj.kernel,
            f"{hf_prefix}.self_attn.v_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            attn.o_proj.kernel,
            f"{hf_prefix}.self_attn.o_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            attn.q_norm.scale, f"{hf_prefix}.self_attn.q_norm.weight"
        )
        loader.port_weight(
            attn.k_norm.scale, f"{hf_prefix}.self_attn.k_norm.weight"
        )

        # MLP
        mlp = keras_layer.mlp
        loader.port_weight(
            mlp.gate_proj.kernel,
            f"{hf_prefix}.mlp.gate_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            mlp.up_proj.kernel,
            f"{hf_prefix}.mlp.up_proj.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            mlp.down_proj.kernel,
            f"{hf_prefix}.mlp.down_proj.weight",
            hook_fn=transpose,
        )

        # Laurel
        laurel = keras_layer.laurel
        loader.port_weight(
            laurel.linear_left.kernel,
            f"{hf_prefix}.laurel.linear_left.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            laurel.linear_right.kernel,
            f"{hf_prefix}.laurel.linear_right.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            laurel.post_laurel_norm.scale,
            f"{hf_prefix}.laurel.post_laurel_norm.weight",
        )

        # AltUp
        altup = keras_layer.altup
        loader.port_weight(
            altup.correction_coefs.kernel,
            f"{hf_prefix}.altup.correction_coefs.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            altup.prediction_coefs.kernel,
            f"{hf_prefix}.altup.prediction_coefs.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            altup.modality_router.kernel,
            f"{hf_prefix}.altup.modality_router.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            altup.router_norm.scale, f"{hf_prefix}.altup.router_norm.weight"
        )
        loader.port_weight(
            altup.correct_output_scale,
            f"{hf_prefix}.altup.correct_output_scale",
        )

        # Per-layer input fusion
        loader.port_weight(
            keras_layer.per_layer_input_gate.kernel,
            f"{hf_prefix}.per_layer_input_gate.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_layer.per_layer_projection.kernel,
            f"{hf_prefix}.per_layer_projection.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_layer.post_per_layer_input_norm.scale,
            f"{hf_prefix}.post_per_layer_input_norm.weight",
        )

    # --- Final Normalization ---
    loader.port_weight(
        keras_variable=backbone.final_norm.scale,
        hf_weight_key="language_model.norm.weight",
    )

    # --- Vision and Audio Towers ---
    # TODO: Implement the weight porting for the vision and audio encoders.
    # This process is similar to the above: iterate through the layers of
    # `backbone.vision_encoder` and `backbone.audio_encoder` and map their
    # weights to the corresponding keys under "vision_tower" and "audio_tower".
    # For example:
    # loader.port_weight(
    #     backbone.vision_encoder.conv_stem.conv.kernel,
    #     "vision_tower.timm_model.conv_stem.conv.weight",
    #     hook_fn=lambda hf_tensor, keras_shape: np.transpose(hf_tensor, [2, 3, 1, 0]) # For Conv2D
    # )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
