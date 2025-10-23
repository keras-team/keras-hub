import json

import numpy as np
from huggingface_hub import hf_hub_download

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.detr.detr_backbone import DETRBackbone

backbone_cls = DETRBackbone


def convert_backbone_config(transformers_config):
    """Convert HuggingFace config to KerasHub DETRBackbone config."""
    resnet = Backbone.from_preset("resnet_50_imagenet", load_weights=False)

    # Return config for DETRBackbone
    return {
        "image_encoder": resnet,
        "hidden_dim": transformers_config["d_model"],
        "num_encoder_layers": transformers_config["encoder_layers"],
        "num_heads": transformers_config["encoder_attention_heads"],
        "intermediate_size": transformers_config["encoder_ffn_dim"],
        "dropout": 0.0,
        "activation": transformers_config["activation_function"],
        "image_shape": (800, 800, 3),
    }


def convert_weights(backbone, loader, transformers_config):
    """Convert DETR backbone weights from HuggingFace to KerasHub.

    Converts weights for: ResNet encoder + input projection + transformer

    Args:
        backbone: DETRBackbone instance
        loader: SafetensorLoader for loading weights
        transformers_config: HuggingFace config dict
    """

    def port_dense(keras_variable, weight_key):
        """Port a dense layer (transpose for Keras format)."""
        loader.port_weight(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            loader.port_weight(keras_variable.bias, f"{weight_key}.bias")

    def port_ln(keras_variable, weight_key):
        """Port a layer normalization layer."""
        loader.port_weight(keras_variable.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{weight_key}.bias")

    def port_conv2d(keras_variable, weight_key):
        """Port a Conv2D layer (transpose for Keras NHWC format)."""
        loader.port_weight(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        if keras_variable.bias is not None:
            loader.port_weight(keras_variable.bias, f"{weight_key}.bias")

    def port_bn(keras_variable, weight_key):
        """Port a batch normalization layer."""
        loader.port_weight(keras_variable.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{weight_key}.bias")
        loader.port_weight(
            keras_variable.moving_mean, f"{weight_key}.running_mean"
        )
        loader.port_weight(
            keras_variable.moving_variance, f"{weight_key}.running_var"
        )

    def port_mha(keras_mha, weight_prefix, num_heads, hidden_dim):
        """Port multi-head attention weights with proper reshaping."""
        # Query projection
        loader.port_weight(
            keras_mha._query_dense.kernel,
            f"{weight_prefix}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._query_dense.bias is not None:
            loader.port_weight(
                keras_mha._query_dense.bias,
                f"{weight_prefix}.q_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Key projection
        loader.port_weight(
            keras_mha._key_dense.kernel,
            f"{weight_prefix}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._key_dense.bias is not None:
            loader.port_weight(
                keras_mha._key_dense.bias,
                f"{weight_prefix}.k_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Value projection
        loader.port_weight(
            keras_mha._value_dense.kernel,
            f"{weight_prefix}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._value_dense.bias is not None:
            loader.port_weight(
                keras_mha._value_dense.bias,
                f"{weight_prefix}.v_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Output projection
        loader.port_weight(
            keras_mha._output_dense.kernel,
            f"{weight_prefix}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        if keras_mha._output_dense.bias is not None:
            loader.port_weight(
                keras_mha._output_dense.bias, f"{weight_prefix}.out_proj.bias"
            )

    # === 1. ResNet Backbone ===
    resnet = backbone.image_encoder
    resnet.trainable = False

    # Stem
    port_conv2d(
        resnet.get_layer("conv1_conv"),
        "model.backbone.conv_encoder.model.conv1",
    )
    port_bn(
        resnet.get_layer("conv1_bn"), "model.backbone.conv_encoder.model.bn1"
    )

    # Stages (ResNet-50: [3, 4, 6, 3] blocks)
    num_blocks = [3, 4, 6, 3]
    for stack_idx in range(4):
        for block_idx in range(num_blocks[stack_idx]):
            keras_prefix = f"stack{stack_idx}_block{block_idx}"
            layer_num = stack_idx + 1
            hf_prefix = (
                f"model.backbone.conv_encoder.model.layer{layer_num}."
                f"{block_idx}"
            )

            # Downsample (only for first block in each stage)
            if block_idx == 0:
                port_conv2d(
                    resnet.get_layer(f"{keras_prefix}_0_conv"),
                    f"{hf_prefix}.downsample.0",
                )
                port_bn(
                    resnet.get_layer(f"{keras_prefix}_0_bn"),
                    f"{hf_prefix}.downsample.1",
                )

            # Bottleneck blocks
            port_conv2d(
                resnet.get_layer(f"{keras_prefix}_1_conv"), f"{hf_prefix}.conv1"
            )
            port_bn(
                resnet.get_layer(f"{keras_prefix}_1_bn"), f"{hf_prefix}.bn1"
            )
            port_conv2d(
                resnet.get_layer(f"{keras_prefix}_2_conv"), f"{hf_prefix}.conv2"
            )
            port_bn(
                resnet.get_layer(f"{keras_prefix}_2_bn"), f"{hf_prefix}.bn2"
            )
            port_conv2d(
                resnet.get_layer(f"{keras_prefix}_3_conv"), f"{hf_prefix}.conv3"
            )
            port_bn(
                resnet.get_layer(f"{keras_prefix}_3_bn"), f"{hf_prefix}.bn3"
            )

    # === 2. Input Projection ===
    # Find input_proj layer in backbone
    for layer in backbone.layers:
        if "input_proj" in layer.name:
            port_conv2d(layer, "model.input_projection")
            break

    # === 3. Transformer Encoder ===
    encoder = backbone.get_layer("encoder")
    num_encoder_layers = len(encoder.encoder_layers)

    hidden_dim = transformers_config["d_model"]
    num_heads = transformers_config["encoder_attention_heads"]

    for i in range(num_encoder_layers):
        keras_layer = encoder.encoder_layers[i]
        hf_prefix = f"model.encoder.layers.{i}"

        # Self-attention
        port_mha(
            keras_layer.attention_layer,
            f"{hf_prefix}.self_attn",
            num_heads,
            hidden_dim,
        )

        # Layer norms
        port_ln(
            keras_layer.attention_layer_norm,
            f"{hf_prefix}.self_attn_layer_norm",
        )
        port_ln(keras_layer.output_layer_norm, f"{hf_prefix}.final_layer_norm")

        # FFN
        port_dense(keras_layer.intermediate_dense, f"{hf_prefix}.fc1")
        port_dense(keras_layer.output_dense, f"{hf_prefix}.fc2")


def convert_head(task, loader, transformers_config):
    """Convert DETR decoder and heads from HuggingFace to KerasHub.

    Converts weights for: query embeddings, decoder layers,
    class head, bbox head

    Args:
        task: DETRObjectDetector instance
        loader: SafetensorLoader for loading weights
        transformers_config: HuggingFace config dict
    """
    # Build decoder layers by running a forward pass
    image_shape = task.backbone.input_shape[1:]
    dummy_input = np.zeros((1,) + image_shape, dtype=np.float32)
    _ = task(dummy_input)

    def port_dense(keras_variable, weight_key):
        """Port a dense layer (transpose for Keras format)."""
        loader.port_weight(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            loader.port_weight(keras_variable.bias, f"{weight_key}.bias")

    def port_ln(keras_variable, weight_key):
        """Port a layer normalization layer."""
        loader.port_weight(keras_variable.gamma, f"{weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{weight_key}.bias")

    def port_mha(keras_mha, weight_prefix, num_heads, hidden_dim):
        """Port multi-head attention weights with proper reshaping."""
        # Query projection
        loader.port_weight(
            keras_mha._query_dense.kernel,
            f"{weight_prefix}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._query_dense.bias is not None:
            loader.port_weight(
                keras_mha._query_dense.bias,
                f"{weight_prefix}.q_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Key projection
        loader.port_weight(
            keras_mha._key_dense.kernel,
            f"{weight_prefix}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._key_dense.bias is not None:
            loader.port_weight(
                keras_mha._key_dense.bias,
                f"{weight_prefix}.k_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Value projection
        loader.port_weight(
            keras_mha._value_dense.kernel,
            f"{weight_prefix}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        if keras_mha._value_dense.bias is not None:
            loader.port_weight(
                keras_mha._value_dense.bias,
                f"{weight_prefix}.v_proj.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )

        # Output projection
        loader.port_weight(
            keras_mha._output_dense.kernel,
            f"{weight_prefix}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        if keras_mha._output_dense.bias is not None:
            loader.port_weight(
                keras_mha._output_dense.bias, f"{weight_prefix}.out_proj.bias"
            )

    # === 1. Query Embeddings ===
    loader.port_weight(
        task.query_embed.query_embed, "model.query_position_embeddings.weight"
    )

    # === 2. Transformer Decoder ===
    decoder = task.decoder
    num_decoder_layers = len(decoder.decoder_layers)
    hidden_dim = transformers_config["d_model"]
    num_heads = transformers_config["decoder_attention_heads"]

    for i in range(num_decoder_layers):
        keras_layer = decoder.decoder_layers[i]
        hf_prefix = f"model.decoder.layers.{i}"

        # Self-attention
        port_mha(
            keras_layer.self_attention,
            f"{hf_prefix}.self_attn",
            num_heads,
            hidden_dim,
        )

        # Encoder-decoder cross-attention
        port_mha(
            keras_layer.encdec_attention,
            f"{hf_prefix}.encoder_attn",
            num_heads,
            hidden_dim,
        )

        # Layer norms
        port_ln(
            keras_layer.self_attention_layer_norm,
            f"{hf_prefix}.self_attn_layer_norm",
        )
        port_ln(
            keras_layer.encdec_attention_layer_norm,
            f"{hf_prefix}.encoder_attn_layer_norm",
        )
        port_ln(keras_layer.output_layer_norm, f"{hf_prefix}.final_layer_norm")

        # FFN
        port_dense(keras_layer.intermediate_dense, f"{hf_prefix}.fc1")
        port_dense(keras_layer.output_dense, f"{hf_prefix}.fc2")

    # Decoder final layer norm
    port_ln(decoder.output_normalization, "model.decoder.layernorm")

    # === 3. Prediction Heads ===
    # Class prediction head
    port_dense(task.class_head, "class_labels_classifier")

    # Bounding box prediction head (3-layer MLP)
    bbox_mlp = task.bbox_head
    for i, dense_layer in enumerate(bbox_mlp.layers_list):
        port_dense(dense_layer, f"bbox_predictor.layers.{i}")


def convert_image_converter(cls, preset, **kwargs):
    """Convert HuggingFace image processor config to KerasHub image converter.

    Args:
        cls: The KerasHub image converter class to instantiate
        preset: Path to the HuggingFace preset
        **kwargs: Additional kwargs to pass to the image converter

    Returns:
        Instance of the image converter class
    """

    config_path = hf_hub_download(
        preset.replace("hf://", ""), "preprocessor_config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    # Extract image preprocessing parameters
    image_mean = config.get("image_mean")
    image_std = config.get("image_std")

    # Convert to KerasHub format: scale = 1/255/std, offset = -mean/std
    scale = [1.0 / 255.0 / std for std in image_std]
    offset = [-mean / std for mean, std in zip(image_mean, image_std)]

    image_size = (800, 800)

    return cls(
        image_size=image_size,
        scale=scale,
        offset=offset,
        **kwargs,
    )
