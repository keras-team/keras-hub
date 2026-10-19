import numpy as np

from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = DebertaV3Backbone


def _validate_config(transformers_config):
    """Raise a clear error for `deberta-v2` configs KerasHub can't represent.

    HF uses the `deberta-v2` model type for both DeBERTa-v2 (e.g.
    `microsoft/deberta-v2-xlarge`) and DeBERTa-v3 checkpoints. Both use shared
    attention keys, bucketed relative positions, layer-normed relative
    embeddings and no absolute position embeddings; DeBERTa-v2 additionally
    has a convolution block, which `DebertaV3Backbone` supports through
    `conv_kernel_size`. Other variants (e.g. randomly configured test models)
    are rejected.
    """
    errors = []
    # HF configs may omit keys or set them to `null`, so fall back to
    # defaults with `or` rather than relying on `dict.get` defaults alone.
    if not transformers_config.get("relative_attention"):
        errors.append("`relative_attention` must be `True`")
    if (transformers_config.get("position_buckets") or -1) <= 0:
        errors.append("`position_buckets` must be > 0")
    if not transformers_config.get("share_att_key"):
        errors.append("`share_att_key` must be `True`")
    if transformers_config.get("position_biased_input", True):
        errors.append("`position_biased_input` must be `False`")
    conv_kernel_size = transformers_config.get("conv_kernel_size") or 0
    if conv_kernel_size > 0 and conv_kernel_size % 2 == 0:
        errors.append("`conv_kernel_size` must be odd")
    norm_rel_ebd = transformers_config.get("norm_rel_ebd") or "none"
    if "layer_norm" not in [x.strip() for x in norm_rel_ebd.split("|")]:
        errors.append("`norm_rel_ebd` must include `'layer_norm'`")
    pos_att_type = transformers_config.get("pos_att_type") or []
    if isinstance(pos_att_type, str):
        pos_att_type = pos_att_type.split("|")
    pos_att_type = {x.strip().lower() for x in pos_att_type}
    if pos_att_type != {"p2c", "c2p"}:
        errors.append("`pos_att_type` must be `'p2c|c2p'`")
    hidden_size = transformers_config["hidden_size"]
    embedding_size = transformers_config.get("embedding_size") or hidden_size
    if embedding_size != hidden_size:
        errors.append("`embedding_size` must equal `hidden_size`")
    if (transformers_config.get("type_vocab_size") or 0) > 0:
        errors.append("`type_vocab_size` must be 0")
    if (transformers_config.get("hidden_act") or "gelu") != "gelu":
        errors.append("`hidden_act` must be `'gelu'`")
    # `DebertaV3Backbone` hard-codes the layer norm epsilon to 1e-7.
    if not np.isclose(transformers_config.get("layer_norm_eps") or 1e-7, 1e-7):
        errors.append("`layer_norm_eps` must be 1e-7")
    # HF buckets relative positions with `max_relative_positions` (falling
    # back to `max_position_embeddings` when < 1); KerasHub always uses
    # `max_position_embeddings`.
    max_relative_positions = transformers_config.get("max_relative_positions")
    if (max_relative_positions or -1) >= 1 and max_relative_positions != (
        transformers_config["max_position_embeddings"]
    ):
        errors.append(
            "`max_relative_positions` must be < 1 or equal "
            "`max_position_embeddings`"
        )
    if errors:
        raise ValueError(
            "This `deberta-v2` checkpoint uses a configuration that is not "
            "supported by `DebertaV3Backbone`. Unsupported settings: "
            + "; ".join(errors)
            + "."
        )


def convert_backbone_config(transformers_config):
    _validate_config(transformers_config)
    config = {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout": transformers_config["hidden_dropout_prob"],
        "max_sequence_length": transformers_config["max_position_embeddings"],
        "bucket_size": transformers_config["position_buckets"],
    }
    conv_kernel_size = transformers_config.get("conv_kernel_size") or 0
    if conv_kernel_size > 0:
        # Original DeBERTa-v2 checkpoints (e.g. `deberta-v2-xlarge`) add a
        # convolution block. HF defaults to `tanh` and 1 group when unset.
        config["conv_kernel_size"] = conv_kernel_size
        conv_act = transformers_config.get("conv_act") or "tanh"
        config["conv_activation"] = conv_act
        config["conv_groups"] = transformers_config.get("conv_groups") or 1
    return config


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

    # Convolution block of the original DeBERTa-v2 checkpoints.
    if backbone.conv_layer is not None:
        conv = backbone.conv_layer
        loader.port_weight(
            keras_variable=conv.conv.kernel,
            hf_weight_key="encoder.conv.conv.weight",
            # PyTorch `(out, in / groups, kernel)` -> Keras
            # `(kernel, in / groups, out)`.
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (2, 1, 0)),
        )
        loader.port_weight(
            keras_variable=conv.conv.bias,
            hf_weight_key="encoder.conv.conv.bias",
        )
        loader.port_weight(
            keras_variable=conv.layer_norm.gamma,
            hf_weight_key="encoder.conv.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=conv.layer_norm.beta,
            hf_weight_key="encoder.conv.LayerNorm.bias",
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
