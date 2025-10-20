import numpy as np

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = Gemma3Backbone


def load_image_converter_config(transformers_config):
    if "vision_config" in transformers_config:
        image_size = transformers_config["vision_config"].get("image_size", 224)
        return {
            "image_size": (image_size, image_size),
            "scale": 1 / 127.5,
            "offset": -1.0,
        }
    else:
        return None


def convert_backbone_config(transformers_config):
    if transformers_config["model_type"] == "gemma3_text":
        image_size = None
        vision_encoder = None
        transformer_config = transformers_config
    else:
        image_size = transformers_config["vision_config"].get("image_size", 224)
        vision_encoder_config = {
            "image_size": image_size,
            "patch_size": transformers_config["vision_config"].get(
                "patch_size", 16
            ),
            "num_heads": transformers_config["vision_config"].get(
                "num_attention_heads", 12
            ),
            "hidden_dim": transformers_config["vision_config"].get(
                "hidden_size", 768
            ),
            "num_layers": transformers_config["vision_config"].get(
                "num_hidden_layers", 12
            ),
            "intermediate_dim": transformers_config["vision_config"].get(
                "intermediate_size", 3072
            ),
            "output_dim": 2560,
            "pool_size": 4,
            "layer_norm_epsilon": transformers_config["vision_config"].get(
                "layer_norm_eps", 1e-6
            ),
        }
        vision_encoder = Gemma3VisionEncoder(**vision_encoder_config)
        transformer_config = transformers_config["text_config"]

    return {
        "vocabulary_size": transformer_config.get(
            "vocab_size", 262144 if vision_encoder is None else 262208
        ),
        "image_size": image_size,
        "num_layers": transformer_config.get("num_hidden_layers", 26),
        "num_query_heads": transformer_config.get("num_attention_heads", 8),
        "num_key_value_heads": transformer_config.get("num_key_value_heads", 4),
        "hidden_dim": transformer_config.get("hidden_size", 2304),
        "intermediate_dim": transformer_config.get("intermediate_size", 9216),
        "head_dim": transformer_config.get("head_dim", 256),
        "use_post_ffw_norm": True,
        "use_post_attention_norm": True,
        "attention_logit_softcap": transformer_config.get(
            "attn_logit_softcap", None
        ),
        "final_logit_softcap": transformer_config.get(
            "final_logit_softcap", None
        ),
        "use_sliding_window_attention": True,
        "query_head_dim_normalize": True,
        "sliding_window_size": transformer_config.get("sliding_window", 4096),
        "local_rope_scaling_factor": 1.0,
        "global_rope_scaling_factor": (
            transformer_config.get("rope_scaling") or {}
        ).get("factor", 1.0),
        "layer_norm_epsilon": transformer_config.get("rms_norm_eps", 1e-6),
        "use_bidirectional_attention": transformer_config.get(
            "use_bidirectional_attention", False
        ),
        "vision_encoder": vision_encoder,
    }


def convert_weights(backbone, loader, transformers_config):
    if transformers_config["model_type"] == "gemma3_text":
        prefix = "model"
    else:
        prefix = "language_model.model"

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


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
