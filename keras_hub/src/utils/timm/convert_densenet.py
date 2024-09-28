import numpy as np

from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone

backbone_cls = DenseNetBackbone


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]

    if timm_architecture == "densenet121":
        stackwise_num_repeats = [6, 12, 24, 16]
    elif timm_architecture == "densenet169":
        stackwise_num_repeats = [6, 12, 32, 32]
    elif timm_architecture == "densenet201":
        stackwise_num_repeats = [6, 12, 48, 32]
    else:
        raise ValueError(
            f"Currently, the architecture {timm_architecture} is not supported."
        )
    return dict(
        stackwise_num_repeats=stackwise_num_repeats,
        compression_ratio=0.5,
        growth_rate=32,
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

    # Stem
    port_conv2d("conv1_conv", "features.conv0")
    port_batch_normalization("conv1_bn", "features.norm0")

    # Stages
    num_stacks = len(backbone.stackwise_num_repeats)
    for stack_index in range(num_stacks):
        for block_idx in range(backbone.stackwise_num_repeats[stack_index]):
            keras_name = f"stack{stack_index+1}_block{block_idx+1}"
            hf_name = (
                f"features.denseblock{stack_index+1}.denselayer{block_idx+1}"
            )
            port_batch_normalization(f"{keras_name}_1_bn", f"{hf_name}.norm1")
            port_conv2d(f"{keras_name}_1_conv", f"{hf_name}.conv1")
            port_batch_normalization(f"{keras_name}_2_bn", f"{hf_name}.norm2")
            port_conv2d(f"{keras_name}_2_conv", f"{hf_name}.conv2")

    for stack_index in range(num_stacks - 1):
        keras_transition_name = f"transition{stack_index+1}"
        hf_transition_name = f"features.transition{stack_index+1}"
        port_batch_normalization(
            f"{keras_transition_name}_bn", f"{hf_transition_name}.norm"
        )
        port_conv2d(
            f"{keras_transition_name}_conv", f"{hf_transition_name}.conv"
        )

    # Post
    port_batch_normalization("bn", "features.norm5")


def convert_head(task, loader, timm_config):
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key="classifier.weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key="classifier.bias",
    )
