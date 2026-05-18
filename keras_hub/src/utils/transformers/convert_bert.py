import numpy as np

from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = BertBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Detect LayerNorm naming convention. Standard BERT uses
    # "gamma/beta", sentence-transformers use "weight/bias".
    # `loader.get_tensor()` handles `bert.` prefix auto-detection
    # via `SafetensorLoader.get_prefixed_key()` suffix matching.
    try:
        loader.get_tensor("embeddings.LayerNorm.gamma")
        ln_gamma, ln_beta = "gamma", "beta"
    except Exception:
        ln_gamma, ln_beta = "weight", "bias"

    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer(
            "position_embedding"
        ).position_embeddings,
        hf_weight_key="embeddings.position_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("segment_embedding").embeddings,
        hf_weight_key="embeddings.token_type_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").beta,
        hf_weight_key=f"embeddings.LayerNorm.{ln_beta}",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").gamma,
        hf_weight_key=f"embeddings.LayerNorm.{ln_gamma}",
    )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Attention blocks
    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"transformer_layer_{i}")
        attn = block._self_attention_layer
        hf_prefix = "encoder.layer."
        # Attention layers
        loader.port_weight(
            keras_variable=attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.query_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.key_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.value_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn.output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        # Attention layer norm.
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.{ln_beta}",
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.{ln_gamma}",
        )
        # MLP layers
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.bias",
        )
        # Output layer norm.
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.{ln_beta}",
        )
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.{ln_gamma}",
        )

    loader.port_weight(
        keras_variable=backbone.get_layer("pooled_dense").kernel,
        hf_weight_key="pooler.dense.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("pooled_dense").bias,
        hf_weight_key="pooler.dense.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    transformers_config = load_json(preset, HF_TOKENIZER_CONFIG_FILE)
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=transformers_config["do_lower_case"],
        **kwargs,
    )
