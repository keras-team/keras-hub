import numpy as np

from keras_hub.src.models.smollm3.smollm3_backbone import SmolLM3Backbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = SmolLM3Backbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_attention_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "layer_norm_epsilon": transformers_config[
            "rms_norm_eps"
        ],  # Using rms_norm_eps as layer_norm_epsilon
        "max_position_embeddings": transformers_config[
            "max_position_embeddings"
        ],
        "rope_theta": transformers_config["rope_theta"],
        # partial_rotary_factor is not explicitly in config.json
        # but is inherited from the default value in the
        # `_compute_default_rope_parameters()` function
        "partial_rotary_factor": 1.0,
        "attention_bias": transformers_config["attention_bias"],
        "attention_dropout": transformers_config["attention_dropout"],
        # Despite the name, no_rope_layers: 1 = HAS RoPE, 0 = NO RoPE
        "rope_layer_enabled_list": [
            bool(x) for x in transformers_config["no_rope_layers"]
        ],
        "layer_types": transformers_config["layer_types"],
        "mlp_bias": transformers_config["mlp_bias"],
    }


def convert_weights(backbone, loader, transformers_config):
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention layers
        ## Query
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.q_proj.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.k_proj.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.v_proj.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer.self_attn.o_proj.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer.mlp.up_proj.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.mlp.down_proj.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.mlp.gate_proj.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer.post_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    backbone.training = False

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    merges = [" ".join(item) for item in merges]

    # Load all special tokens with the exception of "reserved" ones.
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
