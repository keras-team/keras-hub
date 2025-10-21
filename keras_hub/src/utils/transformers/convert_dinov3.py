import numpy as np

from keras_hub.src.models.dinov3.dinov3_backbone import DINOV3Backbone

backbone_cls = DINOV3Backbone


def convert_backbone_config(transformers_config):
    image_size = transformers_config["image_size"]
    return {
        "patch_size": transformers_config["patch_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_heads": transformers_config["num_attention_heads"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "layer_scale_init_value": transformers_config["layerscale_value"],
        "num_register_tokens": transformers_config["num_register_tokens"],
        "use_mask_token": True,
        "hidden_activation": transformers_config["hidden_act"],
        "use_gated_mlp": transformers_config["use_gated_mlp"],
        "use_query_bias": transformers_config["query_bias"],
        "use_key_bias": transformers_config["key_bias"],
        "use_value_bias": transformers_config["value_bias"],
        "use_proj_bias": transformers_config["proj_bias"],
        "use_mlp_bias": transformers_config["mlp_bias"],
        "attention_dropout": transformers_config["attention_dropout"],
        "drop_path_rate": transformers_config["drop_path_rate"],
        "layer_norm_eps": transformers_config["layer_norm_eps"],
        "image_shape": (image_size, image_size, 3),
        "rope_theta": transformers_config["rope_theta"],
        "apply_layernorm": False,
    }


def convert_weights(backbone, loader, transformers_config):
    if not isinstance(backbone, DINOV3Backbone):
        raise ValueError(
            "The provided backbone must be an instance of DINOV3Backbone. "
            f"Received: {type(backbone)}"
        )

    def port_ln(keras_variable, weight_key):
        loader.port_weight(keras_variable.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        loader.port_weight(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            loader.port_weight(keras_variable.bias, f"{weight_key}.bias")

    # Embedding.
    loader.port_weight(
        keras_variable=backbone.embeddings.cls_token,
        hf_weight_key="embeddings.cls_token",
    )
    if backbone.use_mask_token:
        loader.port_weight(
            keras_variable=backbone.embeddings.mask_token,
            hf_weight_key="embeddings.mask_token",
        )
    if backbone.num_register_tokens > 0:
        loader.port_weight(
            keras_variable=backbone.embeddings.register_tokens,
            hf_weight_key="embeddings.register_tokens",
        )
    loader.port_weight(
        keras_variable=backbone.embeddings.patch_embeddings.projection.kernel,
        hf_weight_key="embeddings.patch_embeddings.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.embeddings.patch_embeddings.projection.bias,
        hf_weight_key="embeddings.patch_embeddings.bias",
    )

    # Encoder.
    for i, layer in enumerate(backbone.encoder.layers):
        prefix = f"layer.{i}"
        port_ln(layer.norm1, f"{prefix}.norm1")
        port_dense(layer.attention.query_dense, f"{prefix}.attention.q_proj")
        port_dense(layer.attention.key_dense, f"{prefix}.attention.k_proj")
        port_dense(layer.attention.value_dense, f"{prefix}.attention.v_proj")
        port_dense(layer.attention.output_dense, f"{prefix}.attention.o_proj")

        loader.port_weight(
            keras_variable=layer.layer_scale1.lambda1,
            hf_weight_key=f"{prefix}.layer_scale1.lambda1",
        )
        port_ln(layer.norm2, f"{prefix}.norm2")
        if backbone.use_gated_mlp:
            port_dense(layer.mlp.gate_proj, f"{prefix}.mlp.gate_proj")
            port_dense(layer.mlp.up_proj, f"{prefix}.mlp.up_proj")
            port_dense(layer.mlp.down_proj, f"{prefix}.mlp.down_proj")
        else:
            port_dense(layer.mlp.up_proj, f"{prefix}.mlp.up_proj")
            port_dense(layer.mlp.down_proj, f"{prefix}.mlp.down_proj")
        loader.port_weight(
            keras_variable=layer.layer_scale2.lambda1,
            hf_weight_key=f"{prefix}.layer_scale2.lambda1",
        )

    port_ln(backbone.layernorm, "norm")
