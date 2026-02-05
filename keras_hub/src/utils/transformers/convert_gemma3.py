import numpy as np
from sentencepiece import SentencePieceProcessor

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Gemma3Backbone


def load_image_converter_config(preset, transformers_config):
    if "vision_config" in transformers_config:
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        mean = preprocessor_config["image_mean"]
        std = preprocessor_config["image_std"]
        rescale_factor = preprocessor_config["rescale_factor"]
        offset = [(-m / s) for m, s in zip(mean, std)]
        scale = [(s * rescale_factor) for s in std]
        image_size = transformers_config["vision_config"].get("image_size", 224)
        return {
            "image_size": (image_size, image_size),
            "scale": scale,
            "offset": offset,
        }
    else:
        return None


def convert_backbone_config(transformers_config):
    if transformers_config["model_type"] == "gemma3_text":
        image_size = None
        vision_encoder = None
        transformer_config = transformers_config
    else:
        vision_config = transformers_config["vision_config"]
        image_size = vision_config["image_size"]
        vision_encoder_config = {
            "image_size": image_size,
            "patch_size": vision_config["patch_size"],
            "num_heads": vision_config["num_attention_heads"],
            "hidden_dim": vision_config["hidden_size"],
            "num_layers": vision_config["num_hidden_layers"],
            "intermediate_dim": vision_config["intermediate_size"],
            "output_dim": 2560,
            "pool_size": 4,
            "layer_norm_epsilon": vision_config.get("layer_norm_eps", 1e-6),
        }
        vision_encoder = Gemma3VisionEncoder(**vision_encoder_config)
        transformer_config = transformers_config["text_config"]

    # Extract rope parameters. HuggingFace uses `rope_scaling` for the
    # global rotary embedding. `rope_parameters` is optional and not used
    # by HF for global scaling when `rope_scaling` is None.
    rope_scaling = transformer_config.get("rope_scaling", None)
    rope_params = transformer_config.get("rope_parameters") or {}

    if rope_scaling is not None:
        rope_global_config = rope_scaling or {}
    else:
        rope_global_config = rope_params.get("full_attention", {})

    rope_local_config = rope_params.get("sliding_attention", {})

    # Determine sliding window attention usage from layer_types or config
    sliding_window = transformer_config.get("sliding_window", None)
    layer_types = transformer_config.get("layer_types", [])

    use_sliding_window_attention = sliding_window not in (None, 0) or any(
        lt == "sliding_attention" for lt in layer_types
    )

    # Determine query_head_dim_normalize
    # If query_pre_attn_scalar equals head_dim, then normalize by head_dim
    query_pre_attn_scalar = transformer_config.get(
        "query_pre_attn_scalar", None
    )
    head_dim = transformer_config.get("head_dim")
    if query_pre_attn_scalar is not None and head_dim is not None:
        query_head_dim_normalize = query_pre_attn_scalar == head_dim
    else:
        query_head_dim_normalize = True

    return {
        "vocabulary_size": transformer_config.get(
            "vocab_size", 262144 if vision_encoder is None else 262208
        ),
        "image_size": image_size,
        "num_layers": transformer_config["num_hidden_layers"],
        "num_query_heads": transformer_config.get("num_attention_heads", 8),
        "num_key_value_heads": transformer_config.get("num_key_value_heads", 4),
        "hidden_dim": transformer_config["hidden_size"],
        "intermediate_dim": transformer_config["intermediate_size"],
        "head_dim": transformer_config["head_dim"],
        # Gemma3 models use post-norm and post-attention norm by default
        "use_post_ffw_norm": transformer_config.get("use_post_ffw_norm", True),
        "use_post_attention_norm": transformer_config.get(
            "use_post_attention_norm", True
        ),
        # Handle soft-capping parameters (may be null)
        "attention_logit_soft_cap": transformer_config.get(
            "attn_logit_softcapping", None
        ),
        "final_logit_soft_cap": transformer_config.get(
            "final_logit_softcapping", None
        ),
        # Use sliding window attention if configured
        "use_sliding_window_attention": use_sliding_window_attention,
        # Normalize query by head_dim if query_pre_attn_scalar == head_dim
        "query_head_dim_normalize": query_head_dim_normalize,
        # Sliding window size (default to 1024 for full attention layers)
        "sliding_window_size": sliding_window or 4096,
        # Rope scaling factors for local (sliding) and global (full) attention
        "local_rope_scaling_factor": rope_local_config.get("factor", 1.0),
        "global_rope_scaling_factor": rope_global_config.get("factor", 1.0),
        "layer_norm_epsilon": transformer_config.get("rms_norm_eps", 1e-6),
        "use_bidirectional_attention": transformer_config.get(
            "use_bidirectional_attention", False
        ),
        # Gemma3 uses query/key normalization by default
        "use_query_key_norm": transformer_config.get(
            "use_query_key_norm", True
        ),
        "vision_encoder": vision_encoder,
    }


def convert_weights(backbone, loader, transformers_config):
    if transformers_config["model_type"] == "gemma3_text":
        prefix = "model"
    else:
        prefix = _resolve_multimodal_prefix(loader)

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=f"{prefix}.embed_tokens.weight",
    )

    def transpose(x, shape):
        return np.transpose(x)

    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        image_encoder = vision_encoder.get_layer("image_encoder")

        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.kernel,
            hf_weight_key="vision_tower.vision_model.embeddings.patch_embedding.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.patch_embedding.bias,
            hf_weight_key="vision_tower.vision_model.embeddings.patch_embedding.bias",
        )

        loader.port_weight(
            keras_variable=image_encoder.vision_embeddings.position_embedding.embeddings,
            hf_weight_key="vision_tower.vision_model.embeddings.position_embedding.weight",
        )

        for i in range(image_encoder.num_layers):
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.gamma,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_1.beta,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.query_proj.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.query_proj.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.key_proj.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[
                    i
                ].attn.value_proj.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.value_proj.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].attn.out_proj.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias",
            )

            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.gamma,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].layer_norm_2.beta,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_1.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias",
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.kernel,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                keras_variable=image_encoder.resblocks[i].mlp_dense_2.bias,
                hf_weight_key=f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias",
            )

        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.gamma,
            hf_weight_key="vision_tower.vision_model.post_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=image_encoder.encoder_layer_norm.beta,
            hf_weight_key="vision_tower.vision_model.post_layernorm.bias",
        )

        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_soft_embedding_norm.scale,
            hf_weight_key="multi_modal_projector.mm_soft_emb_norm.weight",
        )

        loader.port_weight(
            keras_variable=vision_encoder.get_layer(
                "vision_output_encoder"
            ).vision_input_projection.kernel,
            hf_weight_key="multi_modal_projector.mm_input_projection_weight",
        )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")

        loader.port_weight(
            keras_variable=decoder_layer.pre_attention_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_attention_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.pre_ffw_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_ffw_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.post_feedforward_layernorm.weight",
        )

        # Attention layers

        ## Query
        loader.port_weight(
            keras_variable=decoder_layer.attention.query_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.query_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.q_norm.weight",
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer.attention.key_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.key_norm.scale,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.k_norm.weight",
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer.attention.value_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer.attention.output_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.self_attn.o_proj.weight",
            # rearrange_patterns="c (a b) -> a b c",
            # rearrange_dims={"a": backbone.num_query_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[2], keras_shape[0], keras_shape[1]),
                ),
                axes=(1, 2, 0),
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer.gating_ffw.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.mlp.gate_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.gating_ffw_2.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.mlp.up_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.ffw_linear.kernel,
            hf_weight_key=f"{prefix}.layers.{i}.mlp.down_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=f"{prefix}.norm.weight",
    )

    return backbone


def _resolve_multimodal_prefix(loader):
    candidates = ["model.language_model", "language_model.model"]
    for candidate in candidates:
        key = f"{candidate}.embed_tokens.weight"
        try:
            loader.get_tensor(key)
            return candidate
        except Exception:
            continue
    return candidates[0]


def convert_tokenizer(cls, preset, **kwargs):
    proto = get_file(preset, "tokenizer.model")
    sp = SentencePieceProcessor()
    if isinstance(proto, bytes):
        sp.LoadFromSerializedProto(proto)
    else:
        sp.load(proto)

    has_vision_tokens = (
        sp.PieceToId("<start_of_image>") != sp.unk_id()
        and sp.PieceToId("<img>") != sp.unk_id()
        and sp.PieceToId("<end_of_image>") != sp.unk_id()
    )

    return cls(proto, has_vision_tokens=has_vision_tokens, **kwargs)
