import re

import numpy as np
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = T5Gemma2Backbone


def load_image_converter_config(preset, transformers_config):
    """Load image converter config from HF preprocessor_config.json."""
    encoder_config = transformers_config.get("encoder", {})
    if "vision_config" not in encoder_config:
        return None
    preprocessor_config = load_json(preset, "processor_config.json")
    preprocessor_config = preprocessor_config["image_processor"]
    mean = preprocessor_config["image_mean"]
    std = preprocessor_config["image_std"]
    rescale_factor = preprocessor_config["rescale_factor"]
    offset = [(-m / s) for m, s in zip(mean, std)]
    scale = [(rescale_factor / s) for s in std]
    image_size = encoder_config["vision_config"]["image_size"]
    return {
        "image_size": (image_size, image_size),
        "scale": scale,
        "offset": offset,
    }


def convert_backbone_config(transformers_config):
    """Convert a HuggingFace T5Gemma2 config to KerasHub backbone config."""
    # T5Gemma2EncoderConfig is Gemma3Config with text params at
    # encoder["text_config"]; decoder is Gemma3TextConfig (flat).
    encoder_config = transformers_config["encoder"]
    enc_text = encoder_config["text_config"]
    decoder_config = transformers_config["decoder"]

    hidden_activation = decoder_config["hidden_activation"]
    if hidden_activation == "gelu_pytorch_tanh":
        hidden_activation = "gelu_approximate"

    # Vision encoder (optional).
    vision_encoder = None
    if "vision_config" in encoder_config:
        from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
            Gemma3VisionEncoder,
        )

        vision_config = encoder_config["vision_config"]
        vision_encoder = Gemma3VisionEncoder(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            num_heads=vision_config["num_attention_heads"],
            hidden_dim=vision_config["hidden_size"],
            num_layers=vision_config["num_hidden_layers"],
            intermediate_dim=vision_config["intermediate_size"],
            output_dim=enc_text["hidden_size"],
            pool_size=int(
                vision_config["image_size"]
                // vision_config["patch_size"]
                // int(encoder_config["mm_tokens_per_image"] ** 0.5)
            ),
            layer_norm_epsilon=vision_config["layer_norm_eps"],
        )

    backbone_config = {
        "vocabulary_size": decoder_config["vocab_size"],
        "encoder_hidden_dim": enc_text["hidden_size"],
        "encoder_intermediate_dim": enc_text["intermediate_size"],
        "encoder_num_layers": enc_text["num_hidden_layers"],
        "encoder_num_attention_heads": enc_text["num_attention_heads"],
        "encoder_num_key_value_heads": enc_text["num_key_value_heads"],
        "encoder_head_dim": enc_text["head_dim"],
        "encoder_layer_types": enc_text["layer_types"],
        "tie_word_embeddings": enc_text["tie_word_embeddings"],
        "decoder_hidden_dim": decoder_config["hidden_size"],
        "decoder_intermediate_dim": decoder_config["intermediate_size"],
        "decoder_num_layers": decoder_config["num_hidden_layers"],
        "decoder_num_attention_heads": decoder_config["num_attention_heads"],
        "decoder_num_key_value_heads": decoder_config["num_key_value_heads"],
        "decoder_head_dim": decoder_config["head_dim"],
        "decoder_layer_types": decoder_config["layer_types"],
        "dropout_rate": decoder_config["dropout_rate"],
        "rms_norm_eps": decoder_config["rms_norm_eps"],
        "query_pre_attn_scalar": decoder_config["query_pre_attn_scalar"],
        "attention_bias": decoder_config["attention_bias"],
        "hidden_activation": hidden_activation,
        "initializer_range": decoder_config["initializer_range"],
        "attention_dropout": decoder_config["attention_dropout"],
        "sliding_window": decoder_config["sliding_window"],
        "cross_attention_hidden_size": enc_text["hidden_size"],
        "attn_logit_softcapping": decoder_config["attn_logit_softcapping"],
        "final_logit_softcapping": decoder_config["final_logit_softcapping"],
        "rope_max_wavelength": (
            decoder_config["rope_parameters"]["sliding_attention"]["rope_theta"]
        ),
        "global_rope_scaling_factor": (
            decoder_config["rope_parameters"]["full_attention"]["factor"]
        ),
        "encoder_rope_max_wavelength": (
            enc_text["rope_parameters"]["sliding_attention"]["rope_theta"]
        ),
        "encoder_global_rope_scaling_factor": (
            enc_text["rope_parameters"]["full_attention"]["factor"]
        ),
        # use_qk_norm may not be in config JSON; default True.
        "use_query_key_norm": enc_text.get("use_qk_norm", True),
        "vision_encoder": vision_encoder,
        "eoi_token_index": transformers_config["eoi_token_index"],
    }
    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """Convert T5Gemma2 weights from HuggingFace to KerasHub."""

    def transpose(x, shape):
        return np.transpose(x)

    # === Vision encoder weights ===
    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        image_encoder = vision_encoder.get_layer("image_encoder")

        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.kernel,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.patch_embedding.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.bias,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.patch_embedding.bias",
        )
        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.position_embedding.embeddings,
            hf_weight_key="encoder.vision_tower.vision_model.embeddings.position_embedding.weight",
        )

        for i in range(image_encoder.num_layers):
            hf_vit = f"encoder.vision_tower.vision_model.encoder.layers.{i}"
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.gamma,
                hf_weight_key=f"{hf_vit}.layer_norm1.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.beta,
                hf_weight_key=f"{hf_vit}.layer_norm1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.query_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.q_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.query_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.q_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.k_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.k_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.value_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.v_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.value_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.v_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.kernel,
                hf_weight_key=f"{hf_vit}.self_attn.out_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.bias,
                hf_weight_key=f"{hf_vit}.self_attn.out_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.gamma,
                hf_weight_key=f"{hf_vit}.layer_norm2.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.beta,
                hf_weight_key=f"{hf_vit}.layer_norm2.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.kernel,
                hf_weight_key=f"{hf_vit}.mlp.fc1.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.bias,
                hf_weight_key=f"{hf_vit}.mlp.fc1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.kernel,
                hf_weight_key=f"{hf_vit}.mlp.fc2.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.bias,
                hf_weight_key=f"{hf_vit}.mlp.fc2.bias",
            )

        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.gamma,
            hf_weight_key="encoder.vision_tower.vision_model.post_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.beta,
            hf_weight_key="encoder.vision_tower.vision_model.post_layernorm.bias",
        )

        # Multi-modal projector.
        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_soft_embedding_norm.scale,
            hf_weight_key="encoder.multi_modal_projector.mm_soft_emb_norm.weight",
        )
        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_input_projection.kernel,
            hf_weight_key="encoder.multi_modal_projector.mm_input_projection_weight",
        )

        # EOI embeddings.
        loader.port_weight(
            keras_variable=backbone.encoder_eoi_embedding,
            hf_weight_key="encoder.embed_tokens.eoi_embedding",
        )
        loader.port_weight(
            keras_variable=backbone.decoder_eoi_embedding,
            hf_weight_key="encoder.embed_tokens.eoi_embedding",
        )

    # === Text encoder weights ===
    # Token embeddings.
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="encoder.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.decoder_token_embedding.embeddings,
        hf_weight_key="encoder.embed_tokens.weight",
    )

    # Encoder (weights under encoder.*).
    loader.port_weight(
        keras_variable=backbone.encoder_norm.scale,
        hf_weight_key="encoder.norm.weight",
    )
    for i in range(backbone.encoder_num_layers):
        layer = backbone.get_layer(f"encoder_layer_{i}")
        hf_prefix = f"encoder.layers.{i}"

        # Self-attention Q/K/V/O projections.
        loader.port_weight(
            keras_variable=layer.self_attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.self_attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.o_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )

        # Q/K normalization (Gemma3-style).
        loader.port_weight(
            keras_variable=layer.self_attn.query_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer.self_attn.key_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.k_norm.weight",
        )

        # MLP.
        loader.port_weight(
            keras_variable=layer.mlp.gate_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.gate_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.up_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.up_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.down_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.down_proj.weight",
            hook_fn=lambda w, s: w.T,
        )

        # Layer norms.
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_feedforward_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_feedforward_layernorm.weight"),
        )

    # Decoder (weights directly under decoder.*).
    loader.port_weight(
        keras_variable=backbone.decoder_norm.scale,
        hf_weight_key="decoder.norm.weight",
    )
    for i in range(backbone.decoder_num_layers):
        layer = backbone.get_layer(f"decoder_layer_{i}")
        hf_prefix = f"decoder.layers.{i}"

        # Merged attention (self+cross uses a single self_attn layer).
        loader.port_weight(
            keras_variable=layer.merged_attn.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.key_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.value_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.output_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.o_proj.weight",
            hook_fn=lambda w, s: w.T.reshape(s),
        )

        # Q/K normalization.
        loader.port_weight(
            keras_variable=layer.merged_attn.query_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=layer.merged_attn.key_norm.scale,
            hf_weight_key=f"{hf_prefix}.self_attn.k_norm.weight",
        )

        # MLP.
        loader.port_weight(
            keras_variable=layer.mlp.gate_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.gate_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.up_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.up_proj.weight",
            hook_fn=lambda w, s: w.T,
        )
        loader.port_weight(
            keras_variable=layer.mlp.down_proj.kernel,
            hf_weight_key=f"{hf_prefix}.mlp.down_proj.weight",
            hook_fn=lambda w, s: w.T,
        )

        # Layer norms (no cross-attn norms — merged into self_attn).
        loader.port_weight(
            keras_variable=layer.pre_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_self_attn_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_self_attn_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.pre_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.pre_feedforward_layernorm.weight"),
        )
        loader.port_weight(
            keras_variable=layer.post_feedforward_layernorm.scale,
            hf_weight_key=(f"{hf_prefix}.post_feedforward_layernorm.weight"),
        )


def _build_sentencepiece_proto(tokenizer_config):
    """Build a serialized SentencePiece proto from a tokenizer.json config.

    Used when `tokenizer.model` is not available in the HF repo (e.g.
    T5Gemma2 repos only ship `tokenizer.json`). Reconstructs a BPE
    SentencePiece model that is byte-for-byte compatible with the
    Gemma-family tokenizer.
    """
    vocab = dict(tokenizer_config["model"]["vocab"])
    merges = list(tokenizer_config["model"]["merges"])

    # Normalise merge format: [["a","b"], ...] → ["a b", ...].
    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(m) for m in merges]

    # Include added / special tokens.
    special_token_strings = set()
    for token_info in tokenizer_config.get("added_tokens", []):
        content = token_info["content"]
        vocab[content] = token_info["id"]
        if token_info.get("special", False):
            special_token_strings.add(content)

    # Map each merge result → rank so we can assign piece scores.
    merge_result_rank = {}
    for rank, rule in enumerate(merges):
        parts = rule.split(" ", 1)
        if len(parts) == 2:
            merge_result_rank[parts[0] + parts[1]] = rank

    model_proto = sp_pb2.ModelProto()

    # Trainer spec.
    ts = model_proto.trainer_spec
    ts.model_type = sp_pb2.TrainerSpec.BPE
    ts.vocab_size = len(vocab)
    ts.byte_fallback = True

    # Normalizer spec – replicate the HF normalizer: Replace(" " → "▁").
    ns = model_proto.normalizer_spec
    ns.name = "identity"
    ns.add_dummy_prefix = False
    ns.escape_whitespaces = True
    ns.remove_extra_whitespaces = False

    # Denormalizer spec (for detokenization).
    dns = model_proto.denormalizer_spec
    dns.add_dummy_prefix = False
    dns.escape_whitespaces = True
    dns.remove_extra_whitespaces = False

    # Byte-fallback regex for <0xNN> tokens.
    _byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

    # Sentinel score for base characters (lower than any merge).
    base_score = -float(len(merges) + 1)

    # Pieces must be ordered by ID (index == ID in SP protos).
    for token_str, _token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
        piece = model_proto.pieces.add()
        piece.piece = token_str

        if token_str == "<unk>":
            piece.type = sp_pb2.ModelProto.SentencePiece.UNKNOWN
            piece.score = 0.0
        elif _byte_re.match(token_str):
            piece.type = sp_pb2.ModelProto.SentencePiece.BYTE
            piece.score = 0.0
        elif token_str in special_token_strings:
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
            piece.score = 0.0
        elif token_str in merge_result_rank:
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
            piece.score = -float(merge_result_rank[token_str])
        else:
            # Base character – not the result of any merge.
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
            piece.score = base_score

    return model_proto.SerializeToString()


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a T5Gemma2 tokenizer.

    Tries to load `tokenizer.model` directly from the preset. If the file
    is not present (T5Gemma2 HF repos only publish `tokenizer.json`),
    the SentencePiece proto is constructed programmatically from
    `tokenizer.json` using `_build_sentencepiece_proto`.
    """
    if check_file_exists(preset, "tokenizer.model"):
        proto = get_file(preset, "tokenizer.model")
        return cls(proto=proto, **kwargs)

    # tokenizer.model not found – build proto from tokenizer.json.
    tokenizer_config = load_json(preset, "tokenizer.json")
    proto_bytes = _build_sentencepiece_proto(tokenizer_config)
    return cls(proto=proto_bytes, **kwargs)
