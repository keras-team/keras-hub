import numpy as np

from keras_hub.src.models.bloom.bloom_backbone import BloomBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = BloomBackbone


def convert_backbone_config(transformers_config):
    hidden_size = transformers_config.get(
        "hidden_size", transformers_config.get("n_embed")
    )
    num_heads = transformers_config.get(
        "n_head", transformers_config.get("num_attention_heads")
    )
    num_layers = transformers_config.get(
        "n_layer", transformers_config.get("num_hidden_layers")
    )
    intermediate_dim = transformers_config.get("n_inner", None)
    if intermediate_dim is None:
        intermediate_dim = hidden_size * 4

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": hidden_size,
        "intermediate_dim": intermediate_dim,
        "dropout": transformers_config.get("hidden_dropout", 0.0),
        "layer_norm_epsilon": transformers_config.get(
            "layer_norm_epsilon", 1e-5
        ),
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="word_embeddings_layernorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="word_embeddings_layernorm.bias",
    )

    num_heads = backbone.num_heads
    hidden_dim = backbone.hidden_dim
    head_dim = hidden_dim // num_heads

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._pre_attention_layernorm.gamma,
            hf_weight_key=f"h.{index}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._pre_attention_layernorm.beta,
            hf_weight_key=f"h.{index}.input_layernorm.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._post_attention_layernorm.gamma,
            hf_weight_key=f"h.{index}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._post_attention_layernorm.beta,
            hf_weight_key=f"h.{index}.post_attention_layernorm.bias",
        )

        # Attention layers

        # Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(
                    hf_tensor.reshape(num_heads, 3, head_dim, hidden_dim)[
                        :, 0, :, :
                    ],
                    (2, 0, 1),
                ),
                keras_shape,
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.bias,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor.reshape(num_heads, 3, head_dim)[:, 0, :],
                keras_shape,
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(
                    hf_tensor.reshape(num_heads, 3, head_dim, hidden_dim)[
                        :, 1, :, :
                    ],
                    (2, 0, 1),
                ),
                keras_shape,
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.bias,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor.reshape(num_heads, 3, head_dim)[:, 1, :],
                keras_shape,
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(
                    hf_tensor.reshape(num_heads, 3, head_dim, hidden_dim)[
                        :, 2, :, :
                    ],
                    (2, 0, 1),
                ),
                keras_shape,
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.bias,
            hf_weight_key=f"h.{index}.self_attention.query_key_value.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor.reshape(num_heads, 3, head_dim)[:, 2, :],
                keras_shape,
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"h.{index}.self_attention.dense.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.bias,
            hf_weight_key=f"h.{index}.self_attention.dense.bias",
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._mlp_intermediate_dense.kernel,
            hf_weight_key=f"h.{index}.mlp.dense_h_to_4h.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._mlp_intermediate_dense.bias,
            hf_weight_key=f"h.{index}.mlp.dense_h_to_4h.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._mlp_output_dense.kernel,
            hf_weight_key=f"h.{index}.mlp.dense_4h_to_h.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._mlp_output_dense.bias,
            hf_weight_key=f"h.{index}.mlp.dense_4h_to_h.bias",
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.gamma,
        hf_weight_key="ln_f.weight",
    )
    loader.port_weight(
        keras_variable=backbone.layer_norm.beta,
        hf_weight_key="ln_f.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(merge) for merge in merges]

    return cls(
        vocabulary=vocab,
        merges=merges,
        **kwargs,
    )
