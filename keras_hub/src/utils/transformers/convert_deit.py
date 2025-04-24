import numpy as np

from keras_hub.src.models.deit.deit_backbone import DeiTBackbone

backbone_cls = DeiTBackbone


def convert_backbone_config(transformers_config):
    image_size = transformers_config["image_size"]
    return {
        "image_shape": (image_size, image_size, 3),
        "patch_size": transformers_config["patch_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "dropout_rate": transformers_config["hidden_dropout_prob"],
        "attention_dropout": transformers_config[
            "attention_probs_dropout_prob"
        ],
        "layer_norm_epsilon": transformers_config["layer_norm_eps"],
    }


def convert_weights(backbone, loader, transformers_config):
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

    loader.port_weight(
        keras_variable=backbone.layers[1].patch_embedding.kernel,
        hf_weight_key="deit.embeddings.patch_embeddings.projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    loader.port_weight(
        backbone.layers[1].patch_embedding.bias,
        "deit.embeddings.patch_embeddings.projection.bias",
    )

    loader.port_weight(
        backbone.layers[1].class_token,
        "deit.embeddings.cls_token",
    )

    loader.port_weight(
        backbone.layers[1].distillation_token,
        "deit.embeddings.distillation_token",
    )

    loader.port_weight(
        backbone.layers[1].position_embedding,
        "deit.embeddings.position_embeddings",
    )

    encoder_layers = backbone.layers[2].encoder_layers
    for i, encoder_block in enumerate(encoder_layers):
        prefix = "deit.encoder.layer"
        num_heads = encoder_block.num_heads
        hidden_dim = encoder_block.hidden_dim

        port_mha(
            encoder_block.mha,
            f"{prefix}.{i}.attention",
            num_heads,
            hidden_dim,
        )
        port_ln(encoder_block.layer_norm_1, f"{prefix}.{i}.layernorm_before")
        port_ln(encoder_block.layer_norm_2, f"{prefix}.{i}.layernorm_after")

        port_dense(encoder_block.mlp.dense, f"{prefix}.{i}.intermediate.dense")
        port_dense(
            encoder_block.output_layer.dense, f"{prefix}.{i}.output.dense"
        )
    port_ln(backbone.layers[2].layer_norm, "deit.layernorm")


def convert_head(task, loader, transformers_config):
    prefix = "cls_classifier."
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key=prefix + "weight",
        hook_fn=lambda x, _: x.T,
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key=prefix + "bias",
    )
