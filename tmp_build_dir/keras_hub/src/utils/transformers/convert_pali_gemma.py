import numpy as np

from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = PaliGemmaBackbone


def convert_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    return {
        "vocabulary_size": transformers_config["image_token_index"],
        "image_size": (
            vision_config["image_size"]
            if "image_size" in vision_config.keys()
            else 224
        ),
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"] * 2,
        "head_dim": text_config["num_image_tokens"],
        "vit_patch_size": vision_config["patch_size"],
        "vit_num_heads": vision_config["num_attention_heads"],
        "vit_hidden_dim": vision_config["hidden_size"],
        "vit_num_layers": vision_config["num_hidden_layers"],
        "vit_intermediate_dim": vision_config["intermediate_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    ############################################################################
    # Image Tower
    ############################################################################
    image_encoder = backbone.vit_encoder.get_layer("image_encoder")

    # Embedding
    loader.port_weight(
        keras_variable=image_encoder.vision_embeddings.patch_embedding.bias,
        hf_weight_key="vision_tower.vision_model.embeddings.patch_embedding.bias",
    )

    loader.port_weight(
        keras_variable=image_encoder.vision_embeddings.patch_embedding.kernel,
        hf_weight_key="vision_tower.vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(2, 3, 1, 0)),
    )

    # Positional Embedding
    loader.port_weight(
        keras_variable=image_encoder.vision_embeddings.position_embedding.embeddings,
        hf_weight_key="vision_tower.vision_model.embeddings.position_embedding.weight",
    )

    # Normalization
    loader.port_weight(
        keras_variable=image_encoder.encoder_layer_norm.gamma,
        hf_weight_key="vision_tower.vision_model.post_layernorm.weight",
    )

    loader.port_weight(
        keras_variable=image_encoder.encoder_layer_norm.beta,
        hf_weight_key="vision_tower.vision_model.post_layernorm.bias",
    )

    # ResBlocks
    for index in range(image_encoder.num_layers):
        block = image_encoder.resblocks[index]

        loader.port_weight(
            keras_variable=block.layer_norm_1.beta,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.layer_norm1.bias",
        )

        loader.port_weight(
            keras_variable=block.layer_norm_1.gamma,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.layer_norm1.weight",
        )

        loader.port_weight(
            keras_variable=block.layer_norm_2.beta,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.layer_norm2.bias",
        )

        loader.port_weight(
            keras_variable=block.layer_norm_2.gamma,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.layer_norm2.weight",
        )

        loader.port_weight(
            keras_variable=block.mlp_dense_1.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.mlp.fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        loader.port_weight(
            keras_variable=block.mlp_dense_1.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.mlp.fc1.bias",
        )

        loader.port_weight(
            keras_variable=block.mlp_dense_2.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.mlp.fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        loader.port_weight(
            keras_variable=block.mlp_dense_2.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.mlp.fc2.bias",
        )

        loader.port_weight(
            keras_variable=block.attn.key_proj.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.k_proj.bias",
        )

        loader.port_weight(
            keras_variable=block.attn.key_proj.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        loader.port_weight(
            keras_variable=block.attn.out_proj.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.out_proj.bias",
        )

        loader.port_weight(
            keras_variable=block.attn.out_proj.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.out_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        loader.port_weight(
            keras_variable=block.attn.query_proj.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.q_proj.bias",
        )

        loader.port_weight(
            keras_variable=block.attn.query_proj.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        loader.port_weight(
            keras_variable=block.attn.value_proj.bias,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.v_proj.bias",
        )

        loader.port_weight(
            keras_variable=block.attn.value_proj.kernel,
            hf_weight_key=f"vision_tower.vision_model.encoder.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Multi Modal Projection
    loader.port_weight(
        keras_variable=backbone.vit_encoder.get_layer(
            "image_classifier"
        ).kernel,
        hf_weight_key="multi_modal_projector.linear.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    loader.port_weight(
        keras_variable=backbone.vit_encoder.get_layer("image_classifier").bias,
        hf_weight_key="multi_modal_projector.linear.bias",
    )

    ############################################################################
    # Language Tower
    ############################################################################
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer.pre_attention_norm.scale,
            hf_weight_key=f"language_model.model.layers.{index}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.pre_ffw_norm.scale,
            hf_weight_key=f"language_model.model.layers.{index}.post_attention_layernorm.weight",
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer.attention.query_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.key_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.value_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.output_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.o_proj.weight",
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
            hf_weight_key=f"language_model.model.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.gating_ffw_2.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.ffw_linear.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="language_model.model.norm.weight",
    )

    # Embedding
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="language_model.model.embed_tokens.weight",
        hook_fn=lambda hf_tensor, keras_shape: hf_tensor[: keras_shape[0]],
    )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
