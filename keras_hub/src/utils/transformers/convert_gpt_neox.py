import numpy as np

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.utils.keras_utils import gelu_approximate
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = GPTNeoXBackbone


def convert_backbone_config(transformers_config):
    hidden_act = transformers_config.get("hidden_act", "gelu")
    if hidden_act == "gelu":
        activation = "gelu"
    elif hidden_act in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        activation = gelu_approximate
    else:
        raise ValueError(f"Unsupported hidden_act: {hidden_act}")

    rotary_pct = transformers_config.get("rotary_pct")
    if rotary_pct is None:
        rotary_pct = transformers_config.get("rope_parameters", {}).get(
            "partial_rotary_factor", 0.25
        )

    rotary_emb_base = transformers_config.get("rotary_emb_base")
    if rotary_emb_base is None:
        rotary_emb_base = transformers_config.get("rope_parameters", {}).get(
            "rope_theta", 10000
        )

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config.get("hidden_dropout", 0.0),
        "tie_word_embeddings": transformers_config.get(
            "tie_word_embeddings", False
        ),
        "activation": activation,
        "rotary_percentage": rotary_pct,
        "rotary_max_wavelength": rotary_emb_base,
        "max_sequence_length": transformers_config["max_position_embeddings"],
        "layer_norm_epsilon": transformers_config["layer_norm_eps"],
    }


def convert_weights(backbone, loader, transformers_config):
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="gpt_neox.embed_in.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.token_embedding.reverse_embeddings,
            hf_weight_key="embed_out.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    hidden_dim = transformers_config["hidden_size"]
    num_heads = transformers_config["num_attention_heads"]
    head_dim = hidden_dim // num_heads

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Norms
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"gpt_neox.layers.{i}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"gpt_neox.layers.{i}.input_layernorm.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"gpt_neox.layers.{i}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"gpt_neox.layers.{i}.post_attention_layernorm.bias",
        )

        # Attention QKV
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._qkv_dense.kernel,
            hf_weight_key=f"gpt_neox.layers.{i}.attention.query_key_value.weight",
            hook_fn=lambda hf_tensor, _: np.reshape(
                np.transpose(hf_tensor), (hidden_dim, num_heads, 3 * head_dim)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._qkv_dense.bias,
            hf_weight_key=f"gpt_neox.layers.{i}.attention.query_key_value.bias",
            hook_fn=lambda hf_tensor, _: np.reshape(
                hf_tensor, (num_heads, 3 * head_dim)
            ),
        )

        # Attention Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"gpt_neox.layers.{i}.attention.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.bias,
            hf_weight_key=f"gpt_neox.layers.{i}.attention.dense.bias",
        )

        # MLP
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
        )

    # Final LayerNorm
    loader.port_weight(
        keras_variable=backbone.layer_norm.gamma,
        hf_weight_key="gpt_neox.final_layer_norm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.layer_norm.beta,
        hf_weight_key="gpt_neox.final_layer_norm.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(merge) for merge in merges]

    return cls(vocabulary=vocab, merges=merges, **kwargs)
