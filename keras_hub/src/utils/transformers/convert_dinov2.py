import numpy as np

from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone

backbone_cls = DINOV2Backbone


def convert_backbone_config(transformers_config):
    model_type = transformers_config["model_type"]
    antialias_in_interpolation = False if model_type == "dinov2" else True
    image_size = transformers_config["image_size"]
    intermediate_dim = int(
        transformers_config["hidden_size"] * transformers_config["mlp_ratio"]
    )
    return {
        "patch_size": transformers_config["patch_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_heads": transformers_config["num_attention_heads"],
        "intermediate_dim": intermediate_dim,
        "layer_scale_init_value": transformers_config["layerscale_value"],
        "num_register_tokens": transformers_config.get(
            "num_register_tokens", 0
        ),
        "use_mask_token": transformers_config.get("use_mask_token", True),
        "use_swiglu_ffn": transformers_config["use_swiglu_ffn"],
        "dropout_rate": transformers_config["hidden_dropout_prob"],
        "drop_path_rate": transformers_config["drop_path_rate"],
        "image_shape": (image_size, image_size, 3),
        "position_embedding_shape": (image_size, image_size),
        "antialias_in_interpolation": antialias_in_interpolation,
        "apply_layernorm": transformers_config.get("apply_layernorm", False),
    }


def convert_weights(backbone, loader, transformers_config):
    if not isinstance(backbone, DINOV2Backbone):
        raise ValueError(
            "The provided backbone must be an instance of DINOV2Backbone. "
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

    def port_mha(keras_variable, weight_key, num_heads, hidden_dim):
        # query
        loader.port_weight(
            keras_variable.query_dense.kernel,
            f"{weight_key}.attention.query.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.query_dense.bias,
            f"{weight_key}.attention.query.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        loader.port_weight(
            keras_variable.key_dense.kernel,
            f"{weight_key}.attention.key.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.key_dense.bias,
            f"{weight_key}.attention.key.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        loader.port_weight(
            keras_variable.value_dense.kernel,
            f"{weight_key}.attention.value.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.value_dense.bias,
            f"{weight_key}.attention.value.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        loader.port_weight(
            keras_variable.output_dense.kernel,
            f"{weight_key}.output.dense.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        loader.port_weight(
            keras_variable.output_dense.bias, f"{weight_key}.output.dense.bias"
        )

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
        keras_variable=backbone.embeddings.position_embeddings,
        hf_weight_key="embeddings.position_embeddings",
    )
    # Interpolate position embeddings to match the image shape.
    backbone.embeddings.interpolated_position_embeddings.assign(
        backbone.embeddings._interpolate_position_embeddings(
            backbone.embeddings.position_embeddings,
            patch_size=backbone.patch_size,
            source_shape=backbone.embeddings.position_embedding_shape,
            target_shape=backbone.image_shape,
            antialias=backbone.embeddings.antialias_in_interpolation,
        )
    )
    loader.port_weight(
        keras_variable=backbone.embeddings.patch_embeddings.projection.kernel,
        hf_weight_key="embeddings.patch_embeddings.projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.embeddings.patch_embeddings.projection.bias,
        hf_weight_key="embeddings.patch_embeddings.projection.bias",
    )

    # Encoder.
    hidden_dim = backbone.hidden_dim
    num_heads = backbone.num_heads
    for i, layer in enumerate(backbone.encoder.layers):
        prefix = f"encoder.layer.{i}"
        port_ln(layer.norm1, f"{prefix}.norm1")
        port_mha(
            layer.attention.attention,
            f"{prefix}.attention",
            num_heads,
            hidden_dim,
        )
        loader.port_weight(
            keras_variable=layer.layer_scale1.lambda1,
            hf_weight_key=f"{prefix}.layer_scale1.lambda1",
        )
        port_ln(layer.norm2, f"{prefix}.norm2")
        if backbone.use_swiglu_ffn:
            port_dense(layer.mlp.weights_in, f"{prefix}.mlp.weights_in")
            port_dense(layer.mlp.weights_out, f"{prefix}.mlp.weights_out")
        else:
            port_dense(layer.mlp.fc1, f"{prefix}.mlp.fc1")
            port_dense(layer.mlp.fc2, f"{prefix}.mlp.fc2")
        loader.port_weight(
            keras_variable=layer.layer_scale2.lambda1,
            hf_weight_key=f"{prefix}.layer_scale2.lambda1",
        )

    port_ln(backbone.layernorm, "layernorm")
