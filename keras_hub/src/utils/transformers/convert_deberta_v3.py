import numpy as np

from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = DebertaV3Backbone


def _validate_config(transformers_config):
    """Raise a clear error for DeBERTa-v2 variants KerasHub can't represent.

    HF uses the `deberta-v2` model type for both DeBERTa-v2 and DeBERTa-v3
    checkpoints. `DebertaV3Backbone` only implements the DeBERTa-v3 flavour
    (shared attention keys, bucketed relative positions, layer-normed
    relative embeddings, no absolute position embeddings, no conv layer).
    """
    errors = []
    if not transformers_config.get("relative_attention", False):
        errors.append("`relative_attention` must be `True`")
    if transformers_config.get("position_buckets", -1) <= 0:
        errors.append("`position_buckets` must be > 0")
    if not transformers_config.get("share_att_key", False):
        errors.append("`share_att_key` must be `True`")
    if transformers_config.get("position_biased_input", True):
        errors.append("`position_biased_input` must be `False`")
    if transformers_config.get("conv_kernel_size", 0) > 0:
        errors.append("`conv_kernel_size` must be 0 (no conv layer)")
    norm_rel_ebd = transformers_config.get("norm_rel_ebd", "none")
    if "layer_norm" not in [x.strip() for x in norm_rel_ebd.split("|")]:
        errors.append("`norm_rel_ebd` must include `'layer_norm'`")
    pos_att_type = transformers_config.get("pos_att_type", [])
    if isinstance(pos_att_type, str):
        pos_att_type = pos_att_type.split("|")
    pos_att_type = {x.strip().lower() for x in pos_att_type}
    if pos_att_type != {"p2c", "c2p"}:
        errors.append("`pos_att_type` must be `'p2c|c2p'`")
    embedding_size = transformers_config.get(
        "embedding_size", transformers_config["hidden_size"]
    )
    if embedding_size != transformers_config["hidden_size"]:
        errors.append("`embedding_size` must equal `hidden_size`")
    if transformers_config.get("type_vocab_size", 0) > 0:
        errors.append("`type_vocab_size` must be 0")
    if transformers_config.get("hidden_act", "gelu") != "gelu":
        errors.append("`hidden_act` must be `'gelu'`")
    if errors:
        raise ValueError(
            "This `deberta-v2` checkpoint uses a configuration that is not "
            "supported by `DebertaV3Backbone` (only DeBERTa-v3 style "
            "checkpoints are supported). Unsupported settings: "
            + "; ".join(errors)
            + "."
        )


def convert_backbone_config(transformers_config):
    _validate_config(transformers_config)
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config["hidden_dropout_prob"],
        "max_sequence_length": transformers_config["max_position_embeddings"],
        "bucket_size": transformers_config["position_buckets"],
    }


def convert_weights(backbone, loader, transformers_config):
    def transpose(hf_tensor, _):
        return np.transpose(hf_tensor, axes=(1, 0))

    def transpose_and_reshape(hf_tensor, keras_shape):
        return np.reshape(np.transpose(hf_tensor), keras_shape)

    def reshape(hf_tensor, keras_shape):
        return np.reshape(hf_tensor, keras_shape)

    # Embeddings. `SafetensorLoader` auto-detects the `deberta.` prefix used
    # by task checkpoints (e.g. `DebertaV2ForSequenceClassification`).
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="embeddings.LayerNorm.bias",
    )

    # Relative position embeddings.
    loader.port_weight(
        keras_variable=backbone.relative_embeddings.rel_embeddings,
        hf_weight_key="encoder.rel_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.relative_embeddings.layer_norm.gamma,
        hf_weight_key="encoder.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.relative_embeddings.layer_norm.beta,
        hf_weight_key="encoder.LayerNorm.bias",
    )

    # Encoder layers.
    for i in range(backbone.num_layers):
        keras_layer = backbone.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )
        attention = keras_layer._self_attention_layer
        hf_prefix = f"encoder.layer.{i}."

        # Q, K, V projections.
        for keras_dense, hf_name in (
            (attention._query_dense, "query_proj"),
            (attention._key_dense, "key_proj"),
            (attention._value_dense, "value_proj"),
        ):
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=f"{hf_prefix}attention.self.{hf_name}.weight",
                hook_fn=transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=keras_dense.bias,
                hf_weight_key=f"{hf_prefix}attention.self.{hf_name}.bias",
                hook_fn=reshape,
            )

        # Attention output.
        loader.port_weight(
            keras_variable=attention._output_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.output.dense.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=attention._output_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.output.dense.bias",
        )
        loader.port_weight(
            keras_variable=keras_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}attention.output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=keras_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}attention.output.LayerNorm.bias",
        )

        # Feedforward.
        loader.port_weight(
            keras_variable=keras_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}intermediate.dense.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=keras_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=keras_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}output.dense.weight",
            hook_fn=transpose,
        )
        loader.port_weight(
            keras_variable=keras_layer._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}output.dense.bias",
        )
        loader.port_weight(
            keras_variable=keras_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=keras_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.bias",
        )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "spm.model"), **kwargs)
