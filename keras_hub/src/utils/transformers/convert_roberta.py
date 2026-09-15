import numpy as np

from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = RobertaBackbone

# RoBERTa reserves the first `padding_idx + 1` position embedding rows
# (padding_idx=1 for RoBERTa) and starts real positions at index 2.
_POSITION_OFFSET = 2


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config["hidden_dropout_prob"],
        "max_sequence_length": transformers_config["max_position_embeddings"]
        - _POSITION_OFFSET,
    }


def transpose_and_reshape(x, shape):
    return np.reshape(np.transpose(x), shape)


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="roberta.embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings.position_embedding.position_embeddings,
        hf_weight_key="roberta.embeddings.position_embeddings.weight",
        hook_fn=lambda x, _: x[_POSITION_OFFSET:, :],
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="roberta.embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="roberta.embeddings.LayerNorm.bias",
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        encoder_layer = backbone.transformer_layers[index]
        hf_prefix = f"roberta.encoder.layer.{index}"

        # Attention layers
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.attention.self.query.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.query_dense.bias,
            hf_weight_key=f"{hf_prefix}.attention.self.query.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.attention.self.key.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.key_dense.bias,
            hf_weight_key=f"{hf_prefix}.attention.self.key.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.attention.self.value.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.value_dense.bias,
            hf_weight_key=f"{hf_prefix}.attention.self.value.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer.output_dense.bias,
            hf_weight_key=f"{hf_prefix}.attention.output.dense.bias",
            hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
        )
        # Attention layer norm.
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.attention.output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.attention.output.LayerNorm.bias",
        )
        # MLP layers
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}.intermediate.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}.intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.output.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}.output.dense.bias",
        )
        # Output layer norm.
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.output.LayerNorm.bias",
        )


def convert_tokenizer(cls, preset, **kwargs):
    vocab_file = get_file(preset, "vocab.json")
    merges_file = get_file(preset, "merges.txt")
    return cls(
        vocabulary=vocab_file,
        merges=merges_file,
        **kwargs,
    )
