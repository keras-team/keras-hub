import numpy as np

from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone

backbone_cls = CSPNetBackbone


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]

    if timm_architecture == "cspdarknet53":
        stem_filters = 32
        stem_kernel_size = 3
        stem_strides = 1
        stackwise_depth = [1, 2, 8, 8, 4]
        stackwise_num_filters = [64, 128, 256, 512, 1024]
        bottle_ratio = (0.5,) + (1.0,)
        block_ratio = (1.0,) + (0.5,)
        expand_ratio = (2.0,) + (1.0,)
        stage_type = "csp"
        block_type = "dark_block"
        down_growth = True
        stackwise_strides = 2
    else:
        raise ValueError(
            f"Currently, the architecture {timm_architecture} is not supported."
        )
    return dict(
        stem_filters=stem_filters,
        stem_kernel_size=stem_kernel_size,
        stem_strides=stem_strides,
        stackwise_depth=stackwise_depth,
        stackwise_num_filters=stackwise_num_filters,
        bottle_ratio=bottle_ratio,
        block_ratio=block_ratio,
        expand_ratio=expand_ratio,
        stage_type=stage_type,
        block_type=block_type,
        stackwise_strides=stackwise_strides,
        down_growth=down_growth,
    )


def convert_weights(backbone, loader, timm_config):
    def port_conv2d(hf_weight_prefix, keras_layer_name):
        loader.port_weight(
            backbone.get_layer(keras_layer_name).kernel,
            hf_weight_key=f"{hf_weight_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

    def port_batch_normalization(hf_weight_prefix, keras_layer_name):
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

    # Stem

    stem_filter = backbone.stem_filters
    if not isinstance(stem_filter, (tuple, list)):
        stem_filter = [stem_filter]

    for i in range(len(stem_filter)):
        port_conv2d(f"stem.conv{i + 1}.conv", f"csp_stem_conv_{i}")
        port_batch_normalization(f"stem.conv{i + 1}.bn", f"csp_stem_bn_{i}")

    # Stages
    stackwise_depth = backbone.stackwise_depth
    stage_type = backbone.stage_type
    block_type = backbone.block_type

    for idx, block in enumerate(stackwise_depth):
        port_conv2d(
            f"stages.{idx}.conv_down.conv",
            f"stage_{idx}_{stage_type}_conv_down_1",
        )
        port_batch_normalization(
            f"stages.{idx}.conv_down.bn", f"stage_{idx}_{stage_type}_bn_1"
        )
        port_conv2d(
            f"stages.{idx}.conv_exp.conv", f"stage_{idx}_{stage_type}_conv_exp"
        )
        port_batch_normalization(
            f"stages.{idx}.conv_exp.bn", f"stage_{idx}_{stage_type}_bn_2"
        )

        for i in range(block):
            port_conv2d(
                f"stages.{idx}.blocks.{i}.conv1.conv",
                f"stage_{idx}_block_{i}_{block_type}_conv_1",
            )
            port_batch_normalization(
                f"stages.{idx}.blocks.{i}.conv1.bn",
                f"stage_{idx}_block_{i}_{block_type}_bn_1",
            )
            port_conv2d(
                f"stages.{idx}.blocks.{i}.conv2.conv",
                f"stage_{idx}_block_{i}_{block_type}_conv_2",
            )
            port_batch_normalization(
                f"stages.{idx}.blocks.{i}.conv2.bn",
                f"stage_{idx}_block_{i}_{block_type}_bn_2",
            )
            if block_type == "bottleneck_block":
                port_conv2d(
                    f"stages.{idx}.blocks.{i}.conv3.conv",
                    f"stage_{idx}_block_{i}_{block_type}_conv_3",
                )
                port_batch_normalization(
                    f"stages.{idx}.blocks.{i}.conv3.bn",
                    f"stage_{idx}_block_{i}_{block_type}_bn_3",
                )

        if stage_type == "csp":
            port_conv2d(
                f"stages.{idx}.conv_transition_b.conv",
                f"stage_{idx}_{stage_type}_conv_transition_b",
            )
            port_batch_normalization(
                f"stages.{idx}.conv_transition_b.bn",
                f"stage_{idx}_{stage_type}_transition_b_bn",
            )
            port_conv2d(
                f"stages.{idx}.conv_transition.conv",
                f"stage_{idx}_{stage_type}_conv_transition",
            )
            port_batch_normalization(
                f"stages.{idx}.conv_transition.bn",
                f"stage_{idx}_{stage_type}_transition_bn",
            )

        else:
            port_conv2d(
                f"stages.{idx}.conv_transition.conv",
                f"stage_{idx}_{stage_type}_conv_transition",
            )
            port_batch_normalization(
                f"stages.{idx}.conv_transition.bn",
                f"stage_{idx}_{stage_type}_transition_bn",
            )


def convert_head(task, loader, timm_config):
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key="head.fc.weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key="head.fc.bias",
    )
