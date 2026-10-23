import numpy as np

from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers import convert_bert

backbone_cls = MPNetBackbone

# MPNet reserves the first `padding_idx + 1` position embedding rows
# (padding_idx=1, like RoBERTa) and starts real positions at index 2.
_POSITION_OFFSET = 2
# The reference implementation always buckets relative positions into 32
# buckets, regardless of `relative_attention_num_buckets`.
_NUM_RELATIVE_ATTENTION_BUCKETS = 32

# Sentence-transformer pipeline settings (pooling, normalization and maximum
# sequence length) live in model-agnostic config files, so reuse the readers.
load_task_config = convert_bert.load_task_config
load_preprocessor_config = convert_bert.load_preprocessor_config


def convert_backbone_config(transformers_config):
    num_buckets = transformers_config.get(
        "relative_attention_num_buckets", _NUM_RELATIVE_ATTENTION_BUCKETS
    )
    if num_buckets != _NUM_RELATIVE_ATTENTION_BUCKETS:
        raise ValueError(
            "MPNet checkpoints with `relative_attention_num_buckets` other "
            f"than {_NUM_RELATIVE_ATTENTION_BUCKETS} are not supported. "
            f"Received: relative_attention_num_buckets={num_buckets}"
        )
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config["hidden_dropout_prob"],
        "max_sequence_length": transformers_config["max_position_embeddings"]
        - _POSITION_OFFSET,
        "layer_norm_epsilon": transformers_config.get("layer_norm_eps", 1e-5),
    }


def transpose_and_reshape(x, shape):
    return np.reshape(np.transpose(x), shape)


def convert_weights(backbone, loader, transformers_config):
    # Embeddings.
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings.position_embedding.position_embeddings,
        hf_weight_key="embeddings.position_embeddings.weight",
        hook_fn=lambda x, _: x[_POSITION_OFFSET:, :],
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="embeddings.LayerNorm.bias",
    )

    # Relative position bias, shared by all transformer layers.
    loader.port_weight(
        keras_variable=backbone.relative_position_bias.embeddings,
        hf_weight_key="encoder.relative_attention_bias.weight",
    )

    # Transformer layers.
    for index in range(backbone.num_layers):
        layer = backbone.transformer_layers[index]
        attention = layer.self_attention_layer
        hf_prefix = f"encoder.layer.{index}"

        # Attention projections.
        for keras_dense, hf_name in (
            (attention.query_dense, "q"),
            (attention.key_dense, "k"),
            (attention.value_dense, "v"),
            (attention.output_dense, "o"),
        ):
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=f"{hf_prefix}.attention.attn.{hf_name}.weight",
                hook_fn=transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=keras_dense.bias,
                hf_weight_key=f"{hf_prefix}.attention.attn.{hf_name}.bias",
                hook_fn=lambda hf_tensor, shape: np.reshape(hf_tensor, shape),
            )

        # Attention layer norm.
        loader.port_weight(
            keras_variable=layer.self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.attention.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.attention.LayerNorm.bias",
        )

        # Feedforward layers.
        loader.port_weight(
            keras_variable=layer.feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}.intermediate.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=layer.feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}.intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=layer.feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.output.dense.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=layer.feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}.output.dense.bias",
        )

        # Feedforward layer norm.
        loader.port_weight(
            keras_variable=layer.feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.output.LayerNorm.bias",
        )

    # `pooler.dense.*` is intentionally not converted. It is unused by
    # sentence-transformers and by the reference classification heads.


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, HF_TOKENIZER_CONFIG_FILE)
    lowercase = tokenizer_config.get("do_lower_case", True)
    # `strip_accents=None` means accents are stripped if and only if the text
    # is lowercased.
    strip_accents = tokenizer_config.get("strip_accents")
    if strip_accents is None:
        strip_accents = lowercase
    return cls(
        vocabulary=get_file(preset, "vocab.txt"),
        lowercase=lowercase,
        strip_accents=strip_accents,
        **kwargs,
    )
