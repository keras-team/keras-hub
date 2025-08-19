import numpy as np

from keras_hub.src.models.falcon import FalconBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = FalconBackbone


def convert_backbone_config(transformers_config):
    if transformers_config.get("multi_query", False):
        num_kv_heads = 1
    else:
        num_kv_heads = transformers_config.get(
            "num_kv_heads", transformers_config["num_attention_heads"]
        )
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_attention_heads": transformers_config["num_attention_heads"],
        "head_dim": transformers_config["hidden_size"]
        // transformers_config["num_attention_heads"],
        "intermediate_dim": transformers_config.get(
            "ffn_hidden_size", 4 * transformers_config["hidden_size"]
        ),
        "num_kv_heads": num_kv_heads,
    }


def convert_weights(backbone, loader, transformers_config):
    hidden_dim = transformers_config["hidden_size"]
    num_attention_heads = transformers_config["num_attention_heads"]
    head_dim = hidden_dim // num_attention_heads
    if transformers_config.get("multi_query", False):
        num_kv_heads = 1
    else:
        num_kv_heads = transformers_config.get(
            "num_kv_heads", num_attention_heads
        )

    # Embeddings
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="word_embeddings.weight",
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Norm layer
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.gamma,
            hf_weight_key=f"h.{i}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.beta,
            hf_weight_key=f"h.{i}.input_layernorm.bias",
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer.attention_layer.output_dense.kernel,
            hf_weight_key=f"h.{i}.self_attention.dense.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention_layer.output_dense.bias,
            hf_weight_key=f"h.{i}.self_attention.dense.bias",
        )

        # Load the combined QKV weight
        hf_qkv_tensor = loader.get_tensor(
            f"h.{i}.self_attention.query_key_value.weight"
        )

        if hf_qkv_tensor.shape[0] != hidden_dim:
            hf_qkv_tensor = np.transpose(hf_qkv_tensor)

        query_output_dim = num_attention_heads * head_dim
        kv_output_dim = num_kv_heads * head_dim
        query_kernel = hf_qkv_tensor[:, :query_output_dim]
        key_kernel = hf_qkv_tensor[
            :, query_output_dim : query_output_dim + kv_output_dim
        ]
        value_kernel = hf_qkv_tensor[:, query_output_dim + kv_output_dim :]
        query_kernel = query_kernel.reshape(
            hidden_dim, num_attention_heads, head_dim
        )
        key_kernel = key_kernel.reshape(hidden_dim, num_kv_heads, head_dim)
        value_kernel = value_kernel.reshape(hidden_dim, num_kv_heads, head_dim)
        decoder_layer.attention_layer.query_dense.kernel.assign(query_kernel)
        decoder_layer.attention_layer.key_dense.kernel.assign(key_kernel)
        decoder_layer.attention_layer.value_dense.kernel.assign(value_kernel)

        # Load the combined QKV bias
        hf_qkv_bias = loader.get_tensor(
            f"h.{i}.self_attention.query_key_value.bias"
        )
        query_bias = hf_qkv_bias[:query_output_dim].reshape(num_attention_heads, head_dim)
        key_bias   = hf_qkv_bias[query_output_dim:query_output_dim+kv_output_dim].reshape(num_kv_heads, head_dim)
        value_bias = hf_qkv_bias[query_output_dim+kv_output_dim:].reshape(num_kv_heads, head_dim)

        decoder_layer.attention_layer.query_dense.bias.assign(query_bias)
        decoder_layer.attention_layer.key_dense.bias.assign(key_bias)
        decoder_layer.attention_layer.value_dense.bias.assign(value_bias)

        # MLP dense layers
        loader.port_weight(
            keras_variable=decoder_layer.dense_h_to_4h.kernel,
            hf_weight_key=f"h.{i}.mlp.dense_h_to_4h.weight",
            hook_fn=lambda x, y: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=decoder_layer.dense_h_to_4h.bias,
            hf_weight_key=f"h.{i}.mlp.dense_h_to_4h.bias",
        )

        loader.port_weight(
            keras_variable=decoder_layer.dense_4h_to_h.kernel,
            hf_weight_key=f"h.{i}.mlp.dense_4h_to_h.weight",
            hook_fn=lambda x, y: np.transpose(x),
        )
        
        loader.port_weight(
            keras_variable=decoder_layer.dense_4h_to_h.bias,
            hf_weight_key=f"h.{i}.mlp.dense_4h_to_h.bias",
        )


    if hasattr(backbone, "final_layernorm"):
        loader.port_weight(
            keras_variable=backbone.final_layernorm.gamma,
            hf_weight_key="ln_f.weight",
        )
        loader.port_weight(
            keras_variable=backbone.final_layernorm.beta,
            hf_weight_key="ln_f.bias",
        )


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_data = load_json(preset, "tokenizer.json")
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"].get("merges", None)
    tokenizer_kwargs = {"vocabulary": vocab}
    if merges is not None:
        tokenizer_kwargs["merges"] = merges
    tokenizer_kwargs.update(kwargs)
    return cls(**tokenizer_kwargs)
