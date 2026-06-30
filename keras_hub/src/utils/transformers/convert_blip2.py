"""Convert BLIP-2 checkpoints from HuggingFace to KerasHub.

BLIP-2 (`Blip2ForConditionalGeneration`) and InstructBLIP bundle several
sub-networks under a single checkpoint:

  - ``vision_model.*``        : EVA-CLIP ViT vision encoder.
  - ``qformer.*``             : Querying Transformer (Q-Former), plus the
                                top-level ``query_tokens`` parameter.
  - ``language_projection.*`` : Q-Former -> language-model projection.
  - ``language_model.*``      : the language model, one of OPT (decoder-only,
                                ``text_config.model_type == "opt"``), Flan-T5
                                (encoder-decoder, ``model_type == "t5"``), or
                                Vicuna/LLaMA (decoder-only, ``"llama"``).

The vision encoder, Q-Former and projection are shared across variants; only
the language-model port differs. The weight correspondences here mirror the
manual transfer functions that previously lived in the checkpoint-conversion
scripts.
"""

import numpy as np

from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_custom_opt import BLIP2CustomOPT
from keras_hub.src.models.blip2.blip2_flan_t5_lm import BLIP2FlanT5
from keras_hub.src.models.blip2.blip2_qformer import BLIP2QFormer
from keras_hub.src.models.blip2.blip2_vicuna import BLIP2Vicuna
from keras_hub.src.models.blip2.blip2_vision_encoder import BLIP2VisionEncoder
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = BLIP2Backbone


# hook factories
# PyTorch stores dense kernels as (out, in); KerasHub stores them as (in, out)
# and reshapes attention kernels to (in, heads, head_dim). These hooks adapt
# the HuggingFace tensors to the target Keras variable shape.


def _transpose(w, shape):
    return w.T


def _reshape(w, shape):
    return w.reshape(shape)


def _transpose_reshape(w, shape):
    return w.T.reshape(shape)


def _patch_kernel(w, shape):
    # HF Conv2D weight (out, in, kh, kw) -> Keras (kh, kw, in, out).
    return np.transpose(w, (2, 3, 1, 0))


def _qkv_kernel(part, hidden_dim):
    """Slice fused qkv weight (3*hidden, hidden) -> chunk, then (in, h, hd)."""

    def hook(w, shape):
        chunk = w[part * hidden_dim : (part + 1) * hidden_dim]
        return chunk.T.reshape(shape)

    return hook


def _qkv_bias(part, hidden_dim):
    """Slice fused qkv bias (3*hidden,) -> chunk, then (heads, head_dim)."""

    def hook(w, shape):
        chunk = w[part * hidden_dim : (part + 1) * hidden_dim]
        return chunk.reshape(shape)

    return hook


# config


def _text_model_type(transformers_config):
    return transformers_config["text_config"]["model_type"]


def _is_instructblip(transformers_config):
    return transformers_config.get("model_type") == "instructblip"


def convert_backbone_config(transformers_config):
    """Build BLIP-2 sub-module objects from a HuggingFace config.

    ``BLIP2Backbone`` is constructed from layer objects (not flat config), so
    we build the vision encoder, Q-Former and language model here and return
    them as constructor kwargs.

    Structural fields are read with direct ``[...]`` indexing so the converter
    fails loudly with a ``KeyError`` if HuggingFace renames or drops a key we
    depend on, instead of silently substituting a stale default.

    The exception is the nested ``vision_config`` and ``qformer_config``:
    HuggingFace serializes a sub-config as a diff against its defaults, and
    BLIP-2 / InstructBLIP set no non-default vision or Q-Former fields, so on
    disk these objects hold only ``model_type``. Their structural keys are
    never present, so they keep ``.get(key, default)`` with the
    ``Blip2VisionConfig`` / ``Blip2QFormerConfig`` default (recovering them
    via the raw dict is impossible without the rejected ``Blip2Config``
    reconstruction). ``text_config`` is serialized in full, so its structural
    fields are indexed directly; only keys HuggingFace prunes there (OPT
    ``dropout`` / ``max_position_embeddings``) or that vary by model revision
    (LLaMA-1 ``rope_theta`` / ``num_key_value_heads``) keep ``.get``.
    """
    is_instruct = _is_instructblip(transformers_config)
    vision_config = transformers_config["vision_config"]
    qformer_config = transformers_config["qformer_config"]
    text_config = transformers_config["text_config"]
    num_query_tokens = transformers_config["num_query_tokens"]

    # vision_config / qformer_config are pruned to `model_type` on disk (see
    # above), so every structural field falls back to its HuggingFace default.
    vision_dim = vision_config.get("hidden_size", 1408)
    vision_encoder = BLIP2VisionEncoder(
        image_size=vision_config.get("image_size", 224),
        patch_size=vision_config.get("patch_size", 14),
        num_layers=vision_config.get("num_hidden_layers", 39),
        num_heads=vision_config.get("num_attention_heads", 16),
        hidden_dim=vision_dim,
        intermediate_dim=vision_config.get("intermediate_size", 6144),
        use_patch_bias=True,
        use_class_token=True,
        use_mha_bias=True,
        use_mlp_bias=True,
        dropout_rate=0.0,
        layer_norm_epsilon=vision_config.get("layer_norm_eps", 1e-6),
        initializer_range=vision_config.get("initializer_range", 1e-10),
    )

    qformer_hidden_dim = qformer_config.get("hidden_size", 768)
    qformer = BLIP2QFormer(
        num_query_tokens=num_query_tokens,
        num_layers=qformer_config.get("num_hidden_layers", 12),
        num_heads=qformer_config.get("num_attention_heads", 12),
        hidden_dim=qformer_hidden_dim,
        intermediate_dim=qformer_config.get("intermediate_size", 3072),
        vision_dim=vision_dim,
        cross_attention_frequency=qformer_config.get(
            "cross_attention_frequency", 2
        ),
        dropout=qformer_config.get("hidden_dropout_prob", 0.1),
        layer_norm_epsilon=qformer_config.get("layer_norm_eps", 1e-12),
        instruction_aware=is_instruct,
        qformer_vocabulary_size=(
            qformer_config.get("vocab_size", 30522) if is_instruct else None
        ),
        max_position_embeddings=qformer_config.get(
            "max_position_embeddings", 512
        ),
    )

    # text_config is serialized in full, so structural fields are indexed
    # directly. `.get` is kept only for keys HuggingFace prunes (OPT dropout /
    # max_position_embeddings) or that vary by model revision (LLaMA-1 omits
    # rope_theta / num_key_value_heads and relies on the architecture defaults).
    model_type = text_config["model_type"]
    if model_type == "llama":
        num_attention_heads = text_config["num_attention_heads"]
        language_model = BLIP2Vicuna(
            vocabulary_size=text_config["vocab_size"],
            num_layers=text_config["num_hidden_layers"],
            num_query_heads=num_attention_heads,
            hidden_dim=text_config["hidden_size"],
            intermediate_dim=text_config["intermediate_size"],
            num_query_tokens=num_query_tokens,
            qformer_hidden_dim=qformer_hidden_dim,
            rope_max_wavelength=text_config.get("rope_theta", 10000.0),
            num_key_value_heads=text_config.get(
                "num_key_value_heads", num_attention_heads
            ),
            layer_norm_epsilon=text_config.get("rms_norm_eps", 1e-6),
        )
    elif model_type == "opt":
        language_model = BLIP2CustomOPT(
            vocabulary_size=text_config["vocab_size"],
            num_layers=text_config["num_hidden_layers"],
            num_heads=text_config["num_attention_heads"],
            hidden_dim=text_config["hidden_size"],
            intermediate_dim=text_config["ffn_dim"],
            num_query_tokens=num_query_tokens,
            dropout=text_config.get("dropout", 0.1),
            max_sequence_length=text_config.get(
                "max_position_embeddings", 2048
            ),
            qformer_hidden_dim=qformer_hidden_dim,
        )
    elif model_type == "t5":
        language_model = BLIP2FlanT5(
            vocabulary_size=text_config["vocab_size"],
            num_layers=text_config["num_layers"],
            num_heads=text_config["num_heads"],
            hidden_dim=text_config["d_model"],
            intermediate_dim=text_config["d_ff"],
            key_value_dim=text_config["d_kv"],
            num_query_tokens=num_query_tokens,
            qformer_hidden_dim=qformer_hidden_dim,
            dropout=text_config.get("dropout_rate", 0.1),
            layer_norm_epsilon=text_config.get("layer_norm_epsilon", 1e-6),
        )
    else:
        raise ValueError(
            "KerasHub's BLIP-2 converter supports `opt`, `t5`, and `llama` "
            "language models; received "
            f"`text_config.model_type='{model_type}'`."
        )

    return {
        "vision_encoder": vision_encoder,
        "qformer": qformer,
        "language_model": language_model,
    }


# weights


def _convert_vision_weights(vision_encoder, loader):
    patch = vision_encoder.get_layer("patching_and_embedding")
    encoder = vision_encoder.get_layer("encoder")

    loader.port_weight(
        keras_variable=patch.class_token,
        hf_weight_key="vision_model.embeddings.class_embedding",
        hook_fn=_reshape,
    )
    loader.port_weight(
        keras_variable=patch.patch_embedding.kernel,
        hf_weight_key="vision_model.embeddings.patch_embedding.weight",
        hook_fn=_patch_kernel,
    )
    loader.port_weight(
        keras_variable=patch.patch_embedding.bias,
        hf_weight_key="vision_model.embeddings.patch_embedding.bias",
    )
    loader.port_weight(
        keras_variable=patch.position_embedding.embeddings,
        hf_weight_key="vision_model.embeddings.position_embedding",
        hook_fn=_reshape,
    )

    hidden_dim = vision_encoder.hidden_dim
    for i, block in enumerate(encoder.encoder_layers):
        prefix = f"vision_model.encoder.layers.{i}"

        loader.port_weight(
            keras_variable=block.layer_norm_1.gamma,
            hf_weight_key=f"{prefix}.layer_norm1.weight",
        )
        loader.port_weight(
            keras_variable=block.layer_norm_1.beta,
            hf_weight_key=f"{prefix}.layer_norm1.bias",
        )

        # Fused qkv -> separate q/k/v dense layers.
        qkv_w = f"{prefix}.self_attn.qkv.weight"
        qkv_b = f"{prefix}.self_attn.qkv.bias"
        for part, dense in enumerate(
            (
                block.mha._query_dense,
                block.mha._key_dense,
                block.mha._value_dense,
            )
        ):
            loader.port_weight(
                keras_variable=dense.kernel,
                hf_weight_key=qkv_w,
                hook_fn=_qkv_kernel(part, hidden_dim),
            )
            loader.port_weight(
                keras_variable=dense.bias,
                hf_weight_key=qkv_b,
                hook_fn=_qkv_bias(part, hidden_dim),
            )

        loader.port_weight(
            keras_variable=block.mha._output_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.projection.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=block.mha._output_dense.bias,
            hf_weight_key=f"{prefix}.self_attn.projection.bias",
        )

        loader.port_weight(
            keras_variable=block.layer_norm_2.gamma,
            hf_weight_key=f"{prefix}.layer_norm2.weight",
        )
        loader.port_weight(
            keras_variable=block.layer_norm_2.beta,
            hf_weight_key=f"{prefix}.layer_norm2.bias",
        )

        loader.port_weight(
            keras_variable=block.mlp.dense_1.kernel,
            hf_weight_key=f"{prefix}.mlp.fc1.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=block.mlp.dense_1.bias,
            hf_weight_key=f"{prefix}.mlp.fc1.bias",
        )
        loader.port_weight(
            keras_variable=block.mlp.dense_2.kernel,
            hf_weight_key=f"{prefix}.mlp.fc2.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=block.mlp.dense_2.bias,
            hf_weight_key=f"{prefix}.mlp.fc2.bias",
        )

    loader.port_weight(
        keras_variable=encoder.layer_norm.gamma,
        hf_weight_key="vision_model.post_layernorm.weight",
    )
    loader.port_weight(
        keras_variable=encoder.layer_norm.beta,
        hf_weight_key="vision_model.post_layernorm.bias",
    )


def _convert_qformer_weights(qformer, loader):
    instruction_aware = getattr(qformer, "instruction_aware", False)
    # InstructBLIP nests the embeddings LayerNorm (and adds word / position
    # embeddings) under `qformer.embeddings`; BLIP-2 keeps a top-level
    # `qformer.layernorm` over the query tokens only.
    ln_prefix = "qformer.embeddings." if instruction_aware else "qformer."

    loader.port_weight(
        keras_variable=qformer.query_tokens,
        hf_weight_key="query_tokens",
    )
    loader.port_weight(
        keras_variable=qformer.layer_norm.gamma,
        hf_weight_key=f"{ln_prefix}layernorm.weight",
    )
    loader.port_weight(
        keras_variable=qformer.layer_norm.beta,
        hf_weight_key=f"{ln_prefix}layernorm.bias",
    )

    if instruction_aware:
        loader.port_weight(
            keras_variable=qformer.text_embeddings.word_embeddings.embeddings,
            hf_weight_key="qformer.embeddings.word_embeddings.weight",
        )
        loader.port_weight(
            keras_variable=(
                qformer.text_embeddings.position_embeddings.embeddings
            ),
            hf_weight_key="qformer.embeddings.position_embeddings.weight",
        )

    def port_attention(keras_attn, hf_prefix):
        mha = keras_attn.attention
        loader.port_weight(
            keras_variable=mha._query_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.query.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=mha._query_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.query.bias",
            hook_fn=_reshape,
        )
        loader.port_weight(
            keras_variable=mha._key_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.key.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=mha._key_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.key.bias",
            hook_fn=_reshape,
        )
        loader.port_weight(
            keras_variable=mha._value_dense.kernel,
            hf_weight_key=f"{hf_prefix}attention.value.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=mha._value_dense.bias,
            hf_weight_key=f"{hf_prefix}attention.value.bias",
            hook_fn=_reshape,
        )
        loader.port_weight(
            keras_variable=mha._output_dense.kernel,
            hf_weight_key=f"{hf_prefix}output.dense.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=mha._output_dense.bias,
            hf_weight_key=f"{hf_prefix}output.dense.bias",
        )
        loader.port_weight(
            keras_variable=keras_attn.layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=keras_attn.layer_norm.beta,
            hf_weight_key=f"{hf_prefix}output.LayerNorm.bias",
        )

    for i, layer in enumerate(qformer.transformer_layers):
        prefix = f"qformer.encoder.layer.{i}."
        port_attention(layer.self_attention, f"{prefix}attention.")
        if layer.has_cross_attention:
            port_attention(layer.cross_attention, f"{prefix}crossattention.")

        loader.port_weight(
            keras_variable=layer.intermediate_dense.kernel,
            hf_weight_key=f"{prefix}intermediate_query.dense.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.intermediate_dense.bias,
            hf_weight_key=f"{prefix}intermediate_query.dense.bias",
        )
        loader.port_weight(
            keras_variable=layer.output_dense.kernel,
            hf_weight_key=f"{prefix}output_query.dense.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.output_dense.bias,
            hf_weight_key=f"{prefix}output_query.dense.bias",
        )
        loader.port_weight(
            keras_variable=layer.output_layer_norm.gamma,
            hf_weight_key=f"{prefix}output_query.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.output_layer_norm.beta,
            hf_weight_key=f"{prefix}output_query.LayerNorm.bias",
        )

        if not instruction_aware:
            continue

        # InstructBLIP's second (instruction-text) feed-forward network.
        loader.port_weight(
            keras_variable=layer.text_intermediate_dense.kernel,
            hf_weight_key=f"{prefix}intermediate.dense.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.text_intermediate_dense.bias,
            hf_weight_key=f"{prefix}intermediate.dense.bias",
        )
        loader.port_weight(
            keras_variable=layer.text_output_dense.kernel,
            hf_weight_key=f"{prefix}output.dense.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.text_output_dense.bias,
            hf_weight_key=f"{prefix}output.dense.bias",
        )
        loader.port_weight(
            keras_variable=layer.text_output_layer_norm.gamma,
            hf_weight_key=f"{prefix}output.LayerNorm.weight",
        )
        loader.port_weight(
            keras_variable=layer.text_output_layer_norm.beta,
            hf_weight_key=f"{prefix}output.LayerNorm.bias",
        )


def _convert_projection_weights(language_projection, loader):
    loader.port_weight(
        keras_variable=language_projection.weights[0],
        hf_weight_key="language_projection.weight",
        hook_fn=_transpose,
    )
    loader.port_weight(
        keras_variable=language_projection.weights[1],
        hf_weight_key="language_projection.bias",
    )


def _convert_opt_weights(opt, loader):
    loader.port_weight(
        keras_variable=opt.embeddings_layer.token_embedding.weights[0],
        hf_weight_key="language_model.model.decoder.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=opt.embeddings_layer.position_embedding.weights[0],
        hf_weight_key="language_model.model.decoder.embed_positions.weight",
    )

    for i in range(opt.num_layers):
        layer = opt.transformer_layers[i]
        prefix = f"language_model.model.decoder.layers.{i}"

        loader.port_weight(
            keras_variable=layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"{prefix}.self_attn_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer._self_attention_layer_norm.beta,
            hf_weight_key=f"{prefix}.self_attn_layer_norm.bias",
        )

        attn = layer._self_attention_layer
        for proj, dense in (
            ("q_proj", attn._query_dense),
            ("k_proj", attn._key_dense),
            ("v_proj", attn._value_dense),
        ):
            loader.port_weight(
                keras_variable=dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.{proj}.weight",
                hook_fn=_transpose_reshape,
            )
            loader.port_weight(
                keras_variable=dense.bias,
                hf_weight_key=f"{prefix}.self_attn.{proj}.bias",
                hook_fn=_reshape,
            )
        loader.port_weight(
            keras_variable=attn._output_dense.kernel,
            hf_weight_key=f"{prefix}.self_attn.out_proj.weight",
            hook_fn=_transpose_reshape,
        )
        loader.port_weight(
            keras_variable=attn._output_dense.bias,
            hf_weight_key=f"{prefix}.self_attn.out_proj.bias",
        )

        loader.port_weight(
            keras_variable=layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"{prefix}.final_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer._feedforward_layer_norm.beta,
            hf_weight_key=f"{prefix}.final_layer_norm.bias",
        )
        loader.port_weight(
            keras_variable=layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{prefix}.fc1.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{prefix}.fc1.bias",
        )
        loader.port_weight(
            keras_variable=layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{prefix}.fc2.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer._feedforward_output_dense.bias,
            hf_weight_key=f"{prefix}.fc2.bias",
        )

    loader.port_weight(
        keras_variable=opt.layer_norm.gamma,
        hf_weight_key="language_model.model.decoder.final_layer_norm.weight",
    )
    loader.port_weight(
        keras_variable=opt.layer_norm.beta,
        hf_weight_key="language_model.model.decoder.final_layer_norm.bias",
    )


def _convert_t5_weights(flan_t5, loader):
    t5 = flan_t5.t5

    loader.port_weight(
        keras_variable=t5.token_embedding.embeddings,
        hf_weight_key="language_model.shared.weight",
    )

    def port_t5_attention(attn, hf_prefix, with_rel_bias=False):
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
        if with_rel_bias and attn.use_relative_attention_bias:
            loader.port_weight(
                keras_variable=attn.relative_attention_bias,
                hf_weight_key=(f"{hf_prefix}.relative_attention_bias.weight"),
            )

    def port_t5_ffn(layer, hf_prefix):
        loader.port_weight(
            keras_variable=layer.input_projector.weights[0],
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wi_0.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.gate_projector.weights[0],
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wi_1.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.output_projector.weights[0],
            hf_weight_key=f"{hf_prefix}.DenseReluDense.wo.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            keras_variable=layer.layer_norm.weight,
            hf_weight_key=f"{hf_prefix}.layer_norm.weight",
        )

    # Encoder.
    for i, enc in enumerate(t5.encoder_transformer_layers):
        ep = f"language_model.encoder.block.{i}.layer"
        port_t5_attention(
            enc.self_attention, f"{ep}.0.SelfAttention", with_rel_bias=True
        )
        loader.port_weight(
            keras_variable=enc.self_attention_layer_norm.weight,
            hf_weight_key=f"{ep}.0.layer_norm.weight",
        )
        port_t5_ffn(enc, f"{ep}.1")
    loader.port_weight(
        keras_variable=t5.encoder_layer_norm.weight,
        hf_weight_key="language_model.encoder.final_layer_norm.weight",
    )

    # Decoder.
    for i, dec in enumerate(t5.decoder_transformer_layers):
        dp = f"language_model.decoder.block.{i}.layer"
        port_t5_attention(
            dec.self_attention, f"{dp}.0.SelfAttention", with_rel_bias=True
        )
        loader.port_weight(
            keras_variable=dec.self_attention_layer_norm.weight,
            hf_weight_key=f"{dp}.0.layer_norm.weight",
        )
        port_t5_attention(dec.cross_attention, f"{dp}.1.EncDecAttention")
        loader.port_weight(
            keras_variable=dec.cross_attention_layer_norm.weight,
            hf_weight_key=f"{dp}.1.layer_norm.weight",
        )
        port_t5_ffn(dec, f"{dp}.2")
    loader.port_weight(
        keras_variable=t5.decoder_layer_norm.weight,
        hf_weight_key="language_model.decoder.final_layer_norm.weight",
    )

    # LM head (separate from shared embedding: tie_word_embeddings=False).
    # Not part of the backbone's functional graph, so build it before porting.
    if not flan_t5.lm_head.built:
        flan_t5.lm_head.build((1, 1, flan_t5.hidden_dim))
    loader.port_weight(
        keras_variable=flan_t5.lm_head.kernel,
        hf_weight_key="language_model.lm_head.weight",
        hook_fn=_transpose,
    )


def _convert_vicuna_weights(vicuna, loader):
    """Port a Vicuna (LLaMA) language model under the `language_model.` prefix.

    Mirrors `convert_llama3.convert_weights`, but the InstructBLIP checkpoint
    stores the LLM under `language_model.*` and Vicuna keeps an untied output
    embedding (`language_model.lm_head.weight`).
    """
    llama = vicuna.llama

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    loader.port_weight(
        keras_variable=llama.get_layer("token_embedding").embeddings,
        hf_weight_key="language_model.model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=llama.get_layer("token_embedding").reverse_embeddings,
        hf_weight_key="language_model.lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    for i in range(llama.num_layers):
        decoder_layer = llama.get_layer(f"transformer_layer_{i}")
        prefix = f"language_model.model.layers.{i}"

        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"{prefix}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"{prefix}.post_attention_layernorm.weight",
        )

        attn = decoder_layer._self_attention_layer
        for proj, dense in (
            ("q_proj", attn._query_dense),
            ("k_proj", attn._key_dense),
            ("v_proj", attn._value_dense),
            ("o_proj", attn._output_dense),
        ):
            loader.port_weight(
                keras_variable=dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.{proj}.weight",
                hook_fn=transpose_and_reshape,
            )

        for proj, dense in (
            ("gate_proj", decoder_layer._feedforward_gate_dense),
            ("up_proj", decoder_layer._feedforward_intermediate_dense),
            ("down_proj", decoder_layer._feedforward_output_dense),
        ):
            loader.port_weight(
                keras_variable=dense.kernel,
                hf_weight_key=f"{prefix}.mlp.{proj}.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )

    loader.port_weight(
        keras_variable=llama.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="language_model.model.norm.weight",
    )


def convert_weights(backbone, loader, transformers_config):
    _convert_vision_weights(backbone.vision_encoder, loader)
    _convert_qformer_weights(backbone.qformer, loader)
    _convert_projection_weights(
        backbone.language_model.language_projection, loader
    )

    model_type = _text_model_type(transformers_config)
    if model_type == "opt":
        _convert_opt_weights(backbone.language_model, loader)
    elif model_type == "llama":
        _convert_vicuna_weights(backbone.language_model, loader)
    else:
        _convert_t5_weights(backbone.language_model, loader)


# tokenizer


def convert_tokenizer(cls, preset, **kwargs):
    """Build the concrete BLIP-2 tokenizer for the checkpoint's LM variant.

    The preprocessor declares a generic ``Tokenizer`` class, so the concrete
    subclass (BPE for OPT, SentencePiece for Flan-T5) is selected here from the
    HuggingFace ``config.json``.
    """
    config = load_json(preset, "config.json")
    model_type = config["text_config"]["model_type"]

    if model_type == "llama":
        from keras_hub.src.models.blip2.blip2_vicuna_tokenizer import (
            BLIP2VicunaTokenizer,
        )

        if check_file_exists(preset, "tokenizer.model"):
            proto = get_file(preset, "tokenizer.model")
        else:
            proto = get_file(preset, "spiece.model")
        return BLIP2VicunaTokenizer(proto=proto, **kwargs)

    if model_type == "opt":
        from keras_hub.src.models.blip2.blip2_opt_tokenizer import (
            BLIP2OPTTokenizer,
        )

        tokenizer_config = load_json(preset, "tokenizer.json")
        vocab = dict(tokenizer_config["model"]["vocab"])
        added_tokens = tokenizer_config.get("added_tokens", [])
        for token in added_tokens:
            vocab[token["content"]] = token["id"]
        merges = [
            " ".join(m) if isinstance(m, list) else m
            for m in tokenizer_config["model"]["merges"]
        ]
        unsplittable = [token["content"] for token in added_tokens]
        return BLIP2OPTTokenizer(
            vocabulary=vocab,
            merges=merges,
            unsplittable_tokens=unsplittable or None,
            **kwargs,
        )

    from keras_hub.src.models.blip2.blip2_flan_t5_tokenizer import (
        BLIP2FlanT5Tokenizer,
    )

    if check_file_exists(preset, "spiece.model"):
        proto = get_file(preset, "spiece.model")
    else:
        proto = get_file(preset, "tokenizer.model")
    return BLIP2FlanT5Tokenizer(proto=proto, **kwargs)


# image converter


def load_preprocessor_config(preset, transformers_config):
    """Inject the InstructBLIP Q-Former (BERT WordPiece) tokenizer.

    InstructBLIP ships a second tokenizer for the instruction-aware Q-Former
    under the ``qformer_tokenizer/`` subdirectory of the HuggingFace checkpoint.
    It is loaded here and passed to the preprocessor constructor; for BLIP-2
    (no Q-Former instruction) this returns nothing.
    """
    if not _is_instructblip(transformers_config):
        return None

    from keras_hub.src.models.blip2.blip2_qformer_tokenizer import (
        BLIP2QFormerTokenizer,
    )

    vocab_path = get_file(preset, "qformer_tokenizer/vocab.txt")
    with open(vocab_path, encoding="utf-8") as f:
        vocabulary = [line.rstrip("\n") for line in f]
    qformer_tokenizer = BLIP2QFormerTokenizer(
        vocabulary=vocabulary, lowercase=True
    )
    return {"qformer_tokenizer": qformer_tokenizer}


def load_image_converter_config(preset, transformers_config):
    """Build BLIP2ImageConverter kwargs from the HF config.

    `BLIP2ImageConverter` bakes in the EVA-CLIP normalization statistics, so
    only the spatial size is read from the checkpoint. The size lives in the
    nested ``vision_config``, which HuggingFace prunes to its defaults on disk,
    so we fall back to the ``Blip2VisionConfig`` default (224) when the key is
    absent.
    """
    image_size = transformers_config["vision_config"].get("image_size", 224)
    return {"image_size": (image_size, image_size)}
