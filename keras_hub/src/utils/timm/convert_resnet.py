import numpy as np

from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone

backbone_cls = ResNetBackbone


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]

    if "resnetv2_" in timm_architecture:
        use_pre_activation = True
    else:
        use_pre_activation = False

    if timm_architecture == "resnet18":
        stackwise_num_blocks = [2, 2, 2, 2]
        block_type = "basic_block"
    elif timm_architecture == "resnet26":
        stackwise_num_blocks = [2, 2, 2, 2]
        block_type = "bottleneck_block"
    elif timm_architecture == "resnet34":
        stackwise_num_blocks = [3, 4, 6, 3]
        block_type = "basic_block"
    elif timm_architecture in ("resnet50", "resnetv2_50"):
        stackwise_num_blocks = [3, 4, 6, 3]
        block_type = "bottleneck_block"
    elif timm_architecture in ("resnet101", "resnetv2_101"):
        stackwise_num_blocks = [3, 4, 23, 3]
        block_type = "bottleneck_block"
    elif timm_architecture in ("resnet152", "resnetv2_152"):
        stackwise_num_blocks = [3, 8, 36, 3]
        block_type = "bottleneck_block"
    else:
        raise ValueError(
            f"Currently, the architecture {timm_architecture} is not supported."
        )

    return dict(
        stackwise_num_filters=[64, 128, 256, 512],
        stackwise_num_blocks=stackwise_num_blocks,
        stackwise_num_strides=[1, 2, 2, 2],
        block_type=block_type,
        use_pre_activation=use_pre_activation,
        input_conv_filters=[64],
        input_conv_kernel_sizes=[7],
    )


def convert_weights(backbone, loader, timm_config):
    def port_conv2d(keras_layer_name, hf_weight_prefix):
        loader.port_weight(
            backbone.get_layer(keras_layer_name).kernel,
            hf_weight_key=f"{hf_weight_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

    def port_batch_normalization(keras_layer_name, hf_weight_prefix):
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

    version = "v1" if not backbone.use_pre_activation else "v2"
    block_type = backbone.block_type

    # Stem
    if version == "v1":
        port_conv2d("conv1_conv", "conv1")
        port_batch_normalization("conv1_bn", "bn1")
    else:
        port_conv2d("conv1_conv", "stem.conv")

    # Stages
    num_stacks = len(backbone.stackwise_num_filters)
    for stack_index in range(num_stacks):
        for block_idx in range(backbone.stackwise_num_blocks[stack_index]):
            if version == "v1":
                keras_name = f"stack{stack_index}_block{block_idx}"
                hf_name = f"layer{stack_index+1}.{block_idx}"
            else:
                keras_name = f"stack{stack_index}_block{block_idx}"
                hf_name = f"stages.{stack_index}.blocks.{block_idx}"

            if version == "v1":
                if block_idx == 0 and (
                    block_type == "bottleneck_block" or stack_index > 0
                ):
                    port_conv2d(
                        f"{keras_name}_0_conv", f"{hf_name}.downsample.0"
                    )
                    port_batch_normalization(
                        f"{keras_name}_0_bn", f"{hf_name}.downsample.1"
                    )
                port_conv2d(f"{keras_name}_1_conv", f"{hf_name}.conv1")
                port_batch_normalization(f"{keras_name}_1_bn", f"{hf_name}.bn1")
                port_conv2d(f"{keras_name}_2_conv", f"{hf_name}.conv2")
                port_batch_normalization(f"{keras_name}_2_bn", f"{hf_name}.bn2")
                if block_type == "bottleneck_block":
                    port_conv2d(f"{keras_name}_3_conv", f"{hf_name}.conv3")
                    port_batch_normalization(
                        f"{keras_name}_3_bn", f"{hf_name}.bn3"
                    )
            else:
                if block_idx == 0 and (
                    block_type == "bottleneck_block" or stack_index > 0
                ):
                    port_conv2d(
                        f"{keras_name}_0_conv", f"{hf_name}.downsample.conv"
                    )
                port_batch_normalization(
                    f"{keras_name}_pre_activation_bn", f"{hf_name}.norm1"
                )
                port_conv2d(f"{keras_name}_1_conv", f"{hf_name}.conv1")
                port_batch_normalization(
                    f"{keras_name}_1_bn", f"{hf_name}.norm2"
                )
                port_conv2d(f"{keras_name}_2_conv", f"{hf_name}.conv2")
                if block_type == "bottleneck_block":
                    port_batch_normalization(
                        f"{keras_name}_2_bn", f"{hf_name}.norm3"
                    )
                    port_conv2d(f"{keras_name}_3_conv", f"{hf_name}.conv3")

    # Post
    if version == "v2":
        port_batch_normalization("post_bn", "norm")


def convert_head(task, loader, timm_config):
    v2 = "resnetv2_" in timm_config["architecture"]
    prefix = "head.fc." if v2 else "fc."
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key=prefix + "weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key=prefix + "bias",
    )
