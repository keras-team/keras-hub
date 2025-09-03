from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = T5GemmaBackbone


def convert_backbone_config(transformers_config):
    """Convert a Hugging Face T5Gemma config to a KerasHub backbone config."""
    encoder_config = transformers_config["encoder"]
    decoder_config = transformers_config["decoder"]

    if decoder_config.get("hidden_activation") == "gelu_pytorch_tanh":
        decoder_config["hidden_activation"] = "gelu_approximate"
    if encoder_config.get("hidden_activation") == "gelu_pytorch_tanh":
        encoder_config["hidden_activation"] = "gelu_approximate"

    backbone_config = {
        "vocabulary_size": decoder_config["vocab_size"],
        "encoder_hidden_dim": encoder_config["hidden_size"],
        "encoder_intermediate_dim": encoder_config["intermediate_size"],
        "encoder_num_layers": encoder_config["num_hidden_layers"],
        "encoder_num_attention_heads": encoder_config["num_attention_heads"],
        "encoder_num_key_value_heads": encoder_config["num_key_value_heads"],
        "encoder_head_dim": encoder_config["head_dim"],
        "encoder_layer_types": encoder_config["layer_types"],
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
        "hidden_activation": decoder_config["hidden_activation"],
        "initializer_range": decoder_config["initializer_range"],
        "attention_dropout": decoder_config["attention_dropout"],
        "sliding_window": decoder_config["sliding_window"],
        "cross_attention_hidden_size": encoder_config["hidden_size"],
        "attn_logit_softcapping": decoder_config["attn_logit_softcapping"],
        "final_logit_softcapping": decoder_config["final_logit_softcapping"],
        "rope_max_wavelength": decoder_config["rope_theta"],
    }
    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """Convert T5Gemma from Hugging Face to KerasHub."""
    # Token embeddings.
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="encoder.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.decoder_token_embedding.embeddings,
        hf_weight_key="decoder.embed_tokens.weight",
    )

    # Encoder.
    loader.port_weight(
        keras_variable=backbone.encoder_norm.scale,
        hf_weight_key="encoder.norm.weight",
    )
    for i in range(backbone.encoder_num_layers):
        layer = backbone.get_layer(f"encoder_layer_{i}")
        hf_prefix = f"encoder.layers.{i}"

        # Self-attention.
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

        # Layer norm.
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.pre_self_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.post_self_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.post_feedforward_layernorm.weight",
        )

    # Decoder.
    loader.port_weight(
        keras_variable=backbone.decoder_norm.scale,
        hf_weight_key="decoder.norm.weight",
    )
    for i in range(backbone.decoder_num_layers):
        layer = backbone.get_layer(f"decoder_layer_{i}")
        hf_prefix = f"decoder.layers.{i}"

        # Self-attention.
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

        # Cross-attention.
        loader.port_weight(
            keras_variable=layer.cross_attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.cross_attn.q_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.cross_attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.cross_attn.k_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.cross_attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.cross_attn.v_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.cross_attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.cross_attn.o_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
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

        # Layer norm.
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.pre_self_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.post_self_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.pre_cross_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.pre_cross_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.post_cross_attn_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.post_cross_attn_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=f"{hf_prefix}.post_feedforward_layernorm.weight",
        )


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a T5Gemma tokenizer."""
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
