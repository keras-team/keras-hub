import numpy as np

from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone

backbone_cls = MobileNetBackbone


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]

    if "mobilenetv3_" in timm_architecture:
        input_activation = "hard_swish"
        output_activation = "hard_swish"

    else:
        input_activation = "relu6"
        output_activation = "relu6"

    if timm_architecture == "mobilenetv3_small_050":
        stackwise_num_blocks = [2, 3, 2, 3]
        stackwise_expansion = [
            [40, 56],
            [64, 144, 144],
            [72, 72],
            [144, 288, 288],
        ]
        stackwise_num_filters = [[16, 16], [24, 24, 24], [24, 24], [48, 48, 48]]
        stackwise_kernel_size = [[3, 3], [5, 5, 5], [5, 5], [5, 5, 5]]
        stackwise_num_strides = [[2, 1], [2, 1, 1], [1, 1], [2, 1, 1]]
        stackwise_se_ratio = [
            [None, None],
            [0.25, 0.25, 0.25],
            [0.25, 0.25],
            [0.25, 0.25, 0.25], 
        ]
        stackwise_activation = [
            ["relu", "relu"],
            ["hard_swish", "hard_swish", "hard_swish"],
            ["hard_swish", "hard_swish"],
            ["hard_swish", "hard_swish", "hard_swish"],
        ]
        stackwise_padding = [[1, 1], [2, 2, 2], [2, 2], [2, 2, 2]]
        output_num_filters = 1024
        input_num_filters = 16
        depthwise_filters = 8
        squeeze_and_excite = 0.5
        last_layer_filter = 288
    else:
        raise ValueError(
            f"Currently, the architecture {timm_architecture} is not supported."
        )

    return dict(
        input_num_filters=input_num_filters,
        input_activation=input_activation,
        depthwise_filters=depthwise_filters,
        squeeze_and_excite=squeeze_and_excite,
        stackwise_num_blocks=stackwise_num_blocks,
        stackwise_expansion=stackwise_expansion,
        stackwise_num_filters=stackwise_num_filters,
        stackwise_kernel_size=stackwise_kernel_size,
        stackwise_num_strides=stackwise_num_strides,
        stackwise_se_ratio=stackwise_se_ratio,
        stackwise_activation=stackwise_activation,
        stackwise_padding=stackwise_padding,
        output_num_filters=output_num_filters,
        output_activation=output_activation,
        last_layer_filter=last_layer_filter,
    )


def convert_weights(backbone, loader, timm_config):
    def port_conv2d(keras_layer_name, hf_weight_prefix):
        print(f"porting weights {hf_weight_prefix} -> {keras_layer_name}")
        loader.port_weight(
            backbone.get_layer(keras_layer_name).kernel,
            hf_weight_key=f"{hf_weight_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

    def port_batch_normalization(keras_layer_name, hf_weight_prefix):
        print(f"porting weights {hf_weight_prefix} -> {keras_layer_name}")
        loader.port_weight(
            backbone.get_layer(keras_layer_name).gamma,
            hf_weight_key=f"{hf_weight_prefix}.weight",
        )
        loader.port_weight(
            backbone.get_layer(keras_layer_name).beta,
            hf_weight_key=f"{hf_weight_prefix}.bias",
        )
        loader.port_weight(
            backbone.get_layer(keras_layer_name).moving_mean,
            hf_weight_key=f"{hf_weight_prefix}.running_mean",
        )
        loader.port_weight(
            backbone.get_layer(keras_layer_name).moving_variance,
            hf_weight_key=f"{hf_weight_prefix}.running_var",
        )
        loader.port_weight(
            backbone.get_layer(keras_layer_name).moving_variance,
            hf_weight_key=f"{hf_weight_prefix}.running_var",
        )
        
    # Stem
    port_conv2d("input_conv", "conv_stem")
    port_batch_normalization("input_batch_norm", "bn1")

    # DepthWise Block  (block 0)
    hf_name = "blocks.0.0"
    keras_name = "block_0_0"

    port_conv2d(f"{keras_name}_conv1", f"{hf_name}.conv_dw")
    port_batch_normalization(f"{keras_name}_bn1", f"{hf_name}.bn1")

    port_conv2d(f"{keras_name}_se_conv_reduce", f"{hf_name}.se.conv_reduce")
    port_conv2d(f"{keras_name}_se_conv_expand", f"{hf_name}.se.conv_expand")

    port_conv2d(f"{keras_name}_conv2", f"{hf_name}.conv_pw")
    port_batch_normalization(f"{keras_name}_bn2", f"{hf_name}.bn2")

    # Stages
    num_stacks = len(backbone.stackwise_num_blocks)
    for block_idx in range(num_stacks):
        for inverted_block in range(backbone.stackwise_num_blocks[block_idx]):
            keras_name = f"block_{block_idx+1}_{inverted_block}"
            hf_name = f"blocks.{block_idx+1}.{inverted_block}"

            # Inverted Residual Block
            port_conv2d(f"{keras_name}_conv1", f"{hf_name}.conv_pw")
            port_batch_normalization(f"{keras_name}_bn1", f"{hf_name}.bn1")
            port_conv2d(f"{keras_name}_conv2", f"{hf_name}.conv_dw")
            port_batch_normalization(f"{keras_name}_bn2", f"{hf_name}.bn2")

            if backbone.stackwise_se_ratio[block_idx][inverted_block]:
                port_conv2d(
                    f"{keras_name}_se_conv_reduce",
                    f"{hf_name}.se.conv_reduce",
                )
                port_conv2d(
                    f"{keras_name}_se_conv_expand",
                    f"{hf_name}.se.conv_expand",
                )

            port_conv2d(f"{keras_name}_conv3", f"{hf_name}.conv_pwl")
            port_batch_normalization(f"{keras_name}_bn3", f"{hf_name}.bn3")

    # ConvBnAct Block
    port_conv2d(f"block_{num_stacks+1}_0_conv", f"blocks.{num_stacks+1}.0.conv")
    port_batch_normalization(
        f"block_{num_stacks+1}_0_bn", f"blocks.{num_stacks+1}.0.bn1"
    )

    port_conv2d("output_conv", "conv_head")
    # if version == "v2":
    # port_batch_normalization("output_batch_norm", "bn2")

def convert_head(task, loader, timm_config):
    prefix = "classifier."
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key=prefix + "weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key=prefix + "bias",
    )
