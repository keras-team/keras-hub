import numpy as np

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = MistralBackbone


def convert_backbone_config(transformers_config):
    # Multimodal Mistral variants (e.g. Ministral 3) nest the text model's
    # hyperparameters under `text_config`.
    if "text_config" in transformers_config:
        transformers_config = transformers_config["text_config"]

    # `transformers_config` is the raw `config.json` dict, not a populated
    # `MistralConfig`, so older checkpoints may omit fields that HF defaults
    # at runtime. We use `.get(...)` for any field that may be missing from
    # disk and read it directly only for ones every Mistral preset sets.
    rope_params = transformers_config.get("rope_parameters") or {}

    # `rope_theta` is top-level in transformers < 5 and nested under
    # `rope_parameters` in 5.x; fall back across both.
    rope_theta = transformers_config.get("rope_theta") or rope_params.get(
        "rope_theta"
    )

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        # Optional / cross-version fields.
        "head_dim": transformers_config.get("head_dim"),
        "tie_word_embeddings": transformers_config.get(
            "tie_word_embeddings", False
        ),
        "sliding_window": transformers_config.get("sliding_window"),
        "rope_max_wavelength": rope_theta,
        # YaRN-specific; only populated when `rope_parameters` is set. Defaults
        # match the no-scaling (linear) regime that base Mistral uses.
        "rope_type": rope_params.get("rope_type", "linear"),
        "rope_scaling_factor": rope_params.get("factor", 1.0),
        "rope_beta_fast": rope_params.get("beta_fast", 32.0),
        "rope_beta_slow": rope_params.get("beta_slow", 1.0),
        "rope_original_max_position_embeddings": rope_params.get(
            "original_max_position_embeddings", 4096
        ),
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.embed_tokens.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )
    # When `tie_word_embeddings=True`, HF does not store `lm_head.weight`
    # separately; the output projection shares `model.embed_tokens.weight`,
    # and `ReversibleEmbedding` in keras-hub does the same.
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.token_embedding.reverse_embeddings,
            hf_weight_key="lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.input_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.post_attention_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="model.norm.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
