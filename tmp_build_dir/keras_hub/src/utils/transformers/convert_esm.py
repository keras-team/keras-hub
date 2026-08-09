import numpy as np

from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = ESMBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config["hidden_dropout_prob"],
        "position_embedding_type": transformers_config[
            "position_embedding_type"
        ],
        "pad_token_id": transformers_config["pad_token_id"],
        "max_sequence_length": transformers_config.get(
            "max_position_embeddings", None
        ),
        "layer_norm_eps": transformers_config.get("layer_norm_eps", 1e-12),
        "use_pre_layer_norm": transformers_config.get(
            "emb_layer_norm_before", False
        ),
        "activation": transformers_config.get("activation", "gelu"),
        "max_wavelength": transformers_config.get("max_wavelength", 10000),
    }


def transpose_and_reshape(x, shape):
    return np.reshape(np.transpose(x), shape)


def convert_weights(backbone, loader, transformers_config):
    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    if transformers_config["position_embedding_type"] == "absolute":
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "position_embedding"
            ).position_embeddings,
            hf_weight_key="embeddings.position_embeddings.weight",
        )
    if transformers_config.get("emb_layer_norm_before", False):
        loader.port_weight(
            keras_variable=backbone.get_layer("emb_layer_norm").gamma,
            hf_weight_key="embeddings.layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.get_layer("emb_layer_norm").beta,
            hf_weight_key="embeddings.layer_norm.bias",
        )

    loader.port_weight(
        keras_variable=backbone.output_layer_norm.gamma,
        hf_weight_key="encoder.emb_layer_norm_after.weight",
    )
    loader.port_weight(
        keras_variable=backbone.output_layer_norm.beta,
        hf_weight_key="encoder.emb_layer_norm_after.bias",
    )

    # Attention blocks
    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"transformer_layer_{i}")
        attn = block.attention_layer
        hf_prefix = "encoder.layer."
        # Attention layers
        loader.port_weight(
            keras_variable=attn.q_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.q_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.k_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.k_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.v_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.v_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.o_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.o_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        # Attention layer norm.
        loader.port_weight(
            keras_variable=block.attention_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.attention.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=block.attention_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.attention.LayerNorm.bias",
        )
        # MLP layers
        loader.port_weight(
            keras_variable=block.feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block.feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=block.feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block.feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.bias",
        )
        # Output layer norm.
        loader.port_weight(
            keras_variable=block.feedforward_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=block.feedforward_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.LayerNorm.bias",
        )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=False,
        **kwargs,
    )
