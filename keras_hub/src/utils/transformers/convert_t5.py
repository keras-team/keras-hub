"""Convert T5 checkpoints from HuggingFace to KerasHub.

T5 (`T5ForConditionalGeneration`, `model_type == "t5"`) is an encoder-decoder
transformer. A checkpoint holds:

  - ``shared.weight``      : token embedding shared by encoder and decoder.
  - ``encoder.block.{i}.*``: encoder blocks. Each block has a self-attention
                             sub-layer (``layer.0``) and a feedforward
                             sub-layer (``layer.1``).
  - ``decoder.block.{i}.*``: decoder blocks. Each block has a self-attention
                             (``layer.0``), a cross-attention (``layer.1``) and
                             a feedforward (``layer.2``) sub-layer.
  - ``{encoder,decoder}.final_layer_norm.weight``: final RMS norms.
  - ``lm_head.weight``     : LM head, only for untied variants (T5 v1.1,
                             FLAN-T5).

Only the first block of each stack owns the relative attention bias. The
weight correspondences here mirror the manual transfer functions that
previously lived in `tools/checkpoint_conversion/convert_t5_checkpoints.py`,
and the Flan-T5 port in `convert_blip2.py`.
"""

from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file

try:
    from safetensors import SafetensorError
except ImportError:
    SafetensorError = KeyError

backbone_cls = T5Backbone


# hook factories
# PyTorch stores dense kernels as (out, in); KerasHub stores them as (in, out).


def _transpose(w, shape):
    return w.T


# config

# `T5MultiHeadAttention` hard-codes the relative attention bucketing, so only
# checkpoints using the (default) values below can be converted.
_RELATIVE_ATTENTION_NUM_BUCKETS = 32
_RELATIVE_ATTENTION_MAX_DISTANCE = 128
_SUPPORTED_ACTIVATIONS = ("relu", "gelu", "silu")


def _parse_feed_forward_proj(feed_forward_proj):
    """Split `feed_forward_proj` into `(activation, use_gated_activation)`.

    The value is either ``"{activation}"`` or ``"gated-{activation}"``, e.g.
    ``"relu"`` (T5) or ``"gated-gelu"`` (T5 v1.1, FLAN-T5).
    """
    parts = feed_forward_proj.split("-")
    use_gated_activation = len(parts) == 2 and parts[0] == "gated"
    activation = parts[-1]
    if (
        len(parts) > 2
        or (len(parts) == 2 and not use_gated_activation)
        or activation not in _SUPPORTED_ACTIVATIONS
    ):
        raise ValueError(
            "`feed_forward_proj` must be one of "
            f"{list(_SUPPORTED_ACTIVATIONS)}, optionally prefixed with "
            f"`'gated-'`. Received: feed_forward_proj={feed_forward_proj}"
        )
    return activation, use_gated_activation


def convert_backbone_config(transformers_config):
    """Build `T5Backbone` kwargs from a HuggingFace config.

    Structural fields are read with direct ``[...]`` indexing so the converter
    fails loudly with a ``KeyError`` if HuggingFace renames or drops a key we
    depend on, instead of silently substituting a stale default.

    The exception is keys that the original T5 configs (e.g.
    ``google-t5/t5-small``) omit entirely: ``feed_forward_proj``,
    ``tie_word_embeddings``, ``num_decoder_layers`` and
    ``relative_attention_max_distance``. These keep
    ``.get(key, default)`` with the ``T5Config`` default.

    ``tie_embedding_weights`` controls both whether the LM head reuses the
    token embedding and whether decoder outputs are scaled by
    ``hidden_dim**-0.5`` before it (see `T5Seq2SeqLM`). HuggingFace only
    scales for the original (tied) T5, not for T5 v1.1 / FLAN-T5. Configs
    written by transformers>=5 always set ``tie_word_embeddings=True`` and
    record the scaling in ``scale_decoder_outputs`` instead, so that key takes
    precedence when present.
    """
    num_layers = transformers_config["num_layers"]
    num_decoder_layers = (
        transformers_config.get("num_decoder_layers") or num_layers
    )
    if num_decoder_layers != num_layers:
        raise ValueError(
            "`T5Backbone` requires the same number of encoder and decoder "
            f"layers. Received: num_layers={num_layers}, "
            f"num_decoder_layers={num_decoder_layers}"
        )

    num_buckets = transformers_config.get(
        "relative_attention_num_buckets", _RELATIVE_ATTENTION_NUM_BUCKETS
    )
    max_distance = transformers_config.get(
        "relative_attention_max_distance", _RELATIVE_ATTENTION_MAX_DISTANCE
    )
    if (
        num_buckets != _RELATIVE_ATTENTION_NUM_BUCKETS
        or max_distance != _RELATIVE_ATTENTION_MAX_DISTANCE
    ):
        raise ValueError(
            "`T5Backbone` only supports `relative_attention_num_buckets="
            f"{_RELATIVE_ATTENTION_NUM_BUCKETS}` and "
            "`relative_attention_max_distance="
            f"{_RELATIVE_ATTENTION_MAX_DISTANCE}`. "
            f"Received: relative_attention_num_buckets={num_buckets}, "
            f"relative_attention_max_distance={max_distance}"
        )

    activation, use_gated_activation = _parse_feed_forward_proj(
        transformers_config.get("feed_forward_proj", "relu")
    )

    if "scale_decoder_outputs" in transformers_config:
        tie_embedding_weights = transformers_config["scale_decoder_outputs"]
    else:
        tie_embedding_weights = transformers_config.get(
            "tie_word_embeddings", True
        )

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": num_layers,
        "num_heads": transformers_config["num_heads"],
        "hidden_dim": transformers_config["d_model"],
        "intermediate_dim": transformers_config["d_ff"],
        "key_value_dim": transformers_config["d_kv"],
        "dropout": transformers_config.get("dropout_rate", 0.1),
        "activation": activation,
        "use_gated_activation": use_gated_activation,
        "layer_norm_epsilon": transformers_config.get(
            "layer_norm_epsilon", 1e-6
        ),
        "tie_embedding_weights": tie_embedding_weights,
    }


# weights


def _port_t5_attention(attn, loader, hf_prefix):
    for proj, name in (
        (attn.query_projector, "q"),
        (attn.key_projector, "k"),
        (attn.value_projector, "v"),
        (attn.output_projector, "o"),
    ):
        loader.port_weight(
            keras_variable=proj.kernel,
            hf_weight_key=f"{hf_prefix}.{name}.weight",
            hook_fn=_transpose,
        )
    # Only the first block of each stack owns the relative attention bias.
    if attn.use_relative_attention_bias:
        loader.port_weight(
            keras_variable=attn.relative_attention_bias,
            hf_weight_key=f"{hf_prefix}.relative_attention_bias.weight",
        )


def _port_t5_ffn(layer, loader, hf_prefix):
    if layer.use_gated_activation:
        # HF computes `act(wi_0(x)) * wi_1(x)`. In KerasHub the activation
        # lives on `input_projector`.
        loader.port_weight(
            keras_variable=layer.input_projector.kernel,
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wi_0.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.gate_projector.kernel,
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wi_1.weight",
            hook_fn=_transpose,
        )
    else:
        loader.port_weight(
            keras_variable=layer.input_projector.kernel,
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wi.weight",
            hook_fn=_transpose,
        )
    loader.port_weight(
        keras_variable=layer.output_projector.kernel,
        hf_weight_key=f"{hf_prefix}.DenseReluDense.wo.weight",
        hook_fn=_transpose,
    )
    loader.port_weight(
        keras_variable=layer.layer_norm.weight,
        hf_weight_key=f"{hf_prefix}.layer_norm.weight",
    )


def _convert_embedding_weights(backbone, loader):
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="shared.weight",
    )


def _convert_encoder_weights(backbone, loader):
    for i, enc in enumerate(backbone.encoder_transformer_layers):
        ep = f"encoder.block.{i}.layer"
        _port_t5_attention(enc.self_attention, loader, f"{ep}.0.SelfAttention")
        loader.port_weight(
            keras_variable=enc.self_attention_layer_norm.weight,
            hf_weight_key=f"{ep}.0.layer_norm.weight",
        )
        _port_t5_ffn(enc, loader, f"{ep}.1")
    loader.port_weight(
        keras_variable=backbone.encoder_layer_norm.weight,
        hf_weight_key="encoder.final_layer_norm.weight",
    )


def _convert_decoder_weights(backbone, loader):
    for i, dec in enumerate(backbone.decoder_transformer_layers):
        dp = f"decoder.block.{i}.layer"
        _port_t5_attention(dec.self_attention, loader, f"{dp}.0.SelfAttention")
        loader.port_weight(
            keras_variable=dec.self_attention_layer_norm.weight,
            hf_weight_key=f"{dp}.0.layer_norm.weight",
        )
        _port_t5_attention(
            dec.cross_attention, loader, f"{dp}.1.EncDecAttention"
        )
        loader.port_weight(
            keras_variable=dec.cross_attention_layer_norm.weight,
            hf_weight_key=f"{dp}.1.layer_norm.weight",
        )
        _port_t5_ffn(dec, loader, f"{dp}.2")
    loader.port_weight(
        keras_variable=backbone.decoder_layer_norm.weight,
        hf_weight_key="decoder.final_layer_norm.weight",
    )


def _convert_lm_head_weights(backbone, loader):
    # T5 v1.1 / FLAN-T5 ship a separate `lm_head.weight`. Checkpoints saved by
    # transformers>=5 may instead tie it to `shared.weight` (without output
    # scaling), in which case it is not serialized.
    try:
        lm_head = loader.get_tensor("lm_head.weight")
    except (KeyError, SafetensorError):
        lm_head = loader.get_tensor("shared.weight")
    backbone.token_embedding.reverse_embeddings.assign(lm_head.T)


def convert_weights(backbone, loader, transformers_config):
    _convert_embedding_weights(backbone, loader)
    _convert_encoder_weights(backbone, loader)
    _convert_decoder_weights(backbone, loader)
    if not backbone.tie_embedding_weights:
        _convert_lm_head_weights(backbone, loader)


# tokenizer


def convert_tokenizer(cls, preset, **kwargs):
    """Build a `T5Tokenizer` from the checkpoint's SentencePiece model.

    The official T5 repos ship ``spiece.model``; some re-uploads only ship
    ``tokenizer.model``.
    """
    if check_file_exists(preset, "spiece.model"):
        proto = get_file(preset, "spiece.model")
    else:
        proto = get_file(preset, "tokenizer.model")
    return cls(proto=proto, **kwargs)
