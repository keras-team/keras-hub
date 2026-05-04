import numpy as np

from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)

backbone_cls = SwinTransformerBackbone


def convert_backbone_config(transformers_config):
    image_size = transformers_config["image_size"]
    return {
        "image_shape": (image_size, image_size, 3),
        "patch_size": transformers_config["patch_size"],
        "embed_dim": transformers_config["embed_dim"],
        "depths": tuple(transformers_config["depths"]),
        "num_heads": tuple(transformers_config["num_heads"]),
        "window_size": transformers_config["window_size"],
        "mlp_ratio": transformers_config["mlp_ratio"],
        "qkv_bias": transformers_config["qkv_bias"],
        "dropout_rate": transformers_config["hidden_dropout_prob"],
        "attention_dropout": transformers_config[
            "attention_probs_dropout_prob"
        ],
        "drop_path": transformers_config["drop_path_rate"],
        "patch_norm": True,
    }


def convert_weights(backbone, loader, transformers_config):
    def port_ln(keras_layer, weight_key):
        loader.port_weight(keras_layer.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_layer.beta, f"{weight_key}.bias")

    def port_dense(keras_layer, weight_key):
        loader.port_weight(
            keras_layer.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_layer.bias is not None:
            loader.port_weight(keras_layer.bias, f"{weight_key}.bias")

    # 1. Patch embedding
    loader.port_weight(
        backbone.patch_embedding.proj.kernel,
        "embeddings.patch_embeddings.projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        backbone.patch_embedding.proj.bias,
        "embeddings.patch_embeddings.projection.bias",
    )
    if backbone.patch_embedding.norm is not None:
        port_ln(backbone.patch_embedding.norm, "embeddings.norm")

    # 2. Stages
    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            prefix = f"encoder.layers.{stage_idx}.blocks.{block_idx}"
            attn_prefix = f"{prefix}.attention.self"

            port_ln(block.norm1, f"{prefix}.layernorm_before")
            port_ln(block.norm2, f"{prefix}.layernorm_after")

            # HF stores Q, K, V as separate weights; combine into one QKV
            q_w = loader.get_tensor(f"{attn_prefix}.query.weight")
            k_w = loader.get_tensor(f"{attn_prefix}.key.weight")
            v_w = loader.get_tensor(f"{attn_prefix}.value.weight")
            block.attn.qkv.kernel.assign(
                np.concatenate([q_w, k_w, v_w], axis=0).T
            )
            if transformers_config.get("qkv_bias", True):
                q_b = loader.get_tensor(f"{attn_prefix}.query.bias")
                k_b = loader.get_tensor(f"{attn_prefix}.key.bias")
                v_b = loader.get_tensor(f"{attn_prefix}.value.bias")
                block.attn.qkv.bias.assign(
                    np.concatenate([q_b, k_b, v_b], axis=0)
                )

            port_dense(
                block.attn.proj, f"{prefix}.attention.output.dense"
            )
            loader.port_weight(
                block.attn.relative_position_bias_table,
                f"{attn_prefix}.relative_position_bias_table",
            )

            port_dense(block.mlp.fc1, f"{prefix}.intermediate.dense")
            port_dense(block.mlp.fc2, f"{prefix}.output.dense")

        if stage.downsample is not None:
            ds_prefix = f"encoder.layers.{stage_idx}.downsample"
            loader.port_weight(
                stage.downsample.reduction.kernel,
                f"{ds_prefix}.reduction.weight",
                hook_fn=lambda x, _: x.T,
            )
            port_ln(stage.downsample.norm, f"{ds_prefix}.norm")

    # 3. Final norm
    port_ln(backbone.norm, "layernorm")
