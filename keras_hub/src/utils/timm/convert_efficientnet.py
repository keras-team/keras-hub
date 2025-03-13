import math

import numpy as np

from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)

backbone_cls = EfficientNetBackbone


VARIANT_MAP = {
    "b0": {
        "stackwise_width_coefficients": [1.0] * 7,
        "stackwise_depth_coefficients": [1.0] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "b1": {
        "stackwise_width_coefficients": [1.0] * 7,
        "stackwise_depth_coefficients": [1.1] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "b2": {
        "stackwise_width_coefficients": [1.1] * 7,
        "stackwise_depth_coefficients": [1.2] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "b3": {
        "stackwise_width_coefficients": [1.2] * 7,
        "stackwise_depth_coefficients": [1.4] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "b4": {
        "stackwise_width_coefficients": [1.4] * 7,
        "stackwise_depth_coefficients": [1.8] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "b5": {
        "stackwise_width_coefficients": [1.6] * 7,
        "stackwise_depth_coefficients": [2.2] * 7,
        "stackwise_squeeze_and_excite_ratios": [0.25] * 7,
    },
    "lite0": {
        "stackwise_width_coefficients": [1.0] * 7,
        "stackwise_depth_coefficients": [1.0] * 7,
        "stackwise_squeeze_and_excite_ratios": [0] * 7,
        "activation": "relu6",
    },
    "el": {
        "stackwise_width_coefficients": [1.2] * 6,
        "stackwise_depth_coefficients": [1.4] * 6,
        "stackwise_kernel_sizes": [3, 3, 3, 5, 5, 5],
        "stackwise_num_repeats": [1, 2, 4, 5, 4, 2],
        "stackwise_input_filters": [32, 24, 32, 48, 96, 144],
        "stackwise_output_filters": [24, 32, 48, 96, 144, 192],
        "stackwise_expansion_ratios": [4, 8, 8, 8, 8, 8],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0] * 6,
        "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
        "stackwise_force_input_filters": [24, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [True] + [False] * 5,
        "activation": "relu",
    },
    "em": {
        "stackwise_width_coefficients": [1.0] * 6,
        "stackwise_depth_coefficients": [1.1] * 6,
        "stackwise_kernel_sizes": [3, 3, 3, 5, 5, 5],
        "stackwise_num_repeats": [1, 2, 4, 5, 4, 2],
        "stackwise_input_filters": [32, 24, 32, 48, 96, 144],
        "stackwise_output_filters": [24, 32, 48, 96, 144, 192],
        "stackwise_expansion_ratios": [4, 8, 8, 8, 8, 8],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0] * 6,
        "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
        "stackwise_force_input_filters": [24, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [True] + [False] * 5,
        "activation": "relu",
    },
    "es": {
        "stackwise_width_coefficients": [1.0] * 6,
        "stackwise_depth_coefficients": [1.0] * 6,
        "stackwise_kernel_sizes": [3, 3, 3, 5, 5, 5],
        "stackwise_num_repeats": [1, 2, 4, 5, 4, 2],
        "stackwise_input_filters": [32, 24, 32, 48, 96, 144],
        "stackwise_output_filters": [24, 32, 48, 96, 144, 192],
        "stackwise_expansion_ratios": [4, 8, 8, 8, 8, 8],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0] * 6,
        "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
        "stackwise_force_input_filters": [24, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [True] + [False] * 5,
        "activation": "relu",
    },
    "rw_m": {
        "stackwise_width_coefficients": [1.2] * 6,
        "stackwise_depth_coefficients": [1.2] * 4 + [1.6] * 2,
        "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
        "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
        "stackwise_output_filters": [24, 48, 64, 128, 160, 272],
        "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
        "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
        "stackwise_force_input_filters": [0, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [False] * 6,
        "activation": "silu",
        "num_features": 1792,
    },
    "rw_s": {
        "stackwise_width_coefficients": [1.0] * 6,
        "stackwise_depth_coefficients": [1.0] * 6,
        "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
        "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
        "stackwise_output_filters": [24, 48, 64, 128, 160, 272],
        "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
        "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
        "stackwise_force_input_filters": [0, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [False] * 6,
        "activation": "silu",
        "num_features": 1792,
    },
    "rw_t": {
        "stackwise_width_coefficients": [0.8] * 6,
        "stackwise_depth_coefficients": [0.9] * 6,
        "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
        "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
        "stackwise_output_filters": [24, 48, 64, 128, 160, 256],
        "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
        "stackwise_strides": [1, 2, 2, 2, 1, 2],
        "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
        "stackwise_block_types": ["cba"] + ["fused"] * 2 + ["unfused"] * 3,
        "stackwise_force_input_filters": [0, 0, 0, 0, 0, 0],
        "stackwise_nores_option": [False] * 6,
        "activation": "silu",
    },
}


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]

    base_kwargs = {
        "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
        "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
        "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
        "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
        "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
        "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
        "stackwise_block_types": ["v1"] * 7,
        "min_depth": None,
        "include_stem_padding": True,
        "use_depth_divisor_as_min_depth": True,
        "cap_round_filter_decrease": True,
        "stem_conv_padding": "valid",
        "batch_norm_momentum": 0.9,
        "batch_norm_epsilon": 1e-5,
        "dropout": 0,
        "projection_activation": None,
    }

    variant = "_".join(timm_architecture.split("_")[1:])

    if variant not in VARIANT_MAP:
        raise ValueError(
            f"Currently, the architecture {timm_architecture} is not supported."
        )

    base_kwargs.update(VARIANT_MAP[variant])

    return base_kwargs


def convert_weights(backbone, loader, timm_config):
    timm_architecture = timm_config["architecture"]
    variant = "_".join(timm_architecture.split("_")[1:])

    def port_conv2d(keras_layer, hf_weight_prefix, port_bias=True):
        loader.port_weight(
            keras_layer.kernel,
            hf_weight_key=f"{hf_weight_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

        if port_bias:
            loader.port_weight(
                keras_layer.bias,
                hf_weight_key=f"{hf_weight_prefix}.bias",
            )

    def port_depthwise_conv2d(
        keras_layer,
        hf_weight_prefix,
        port_bias=True,
        depth_multiplier=1,
    ):
        def convert_pt_conv2d_kernel(pt_kernel):
            out_channels, in_channels_per_group, height, width = pt_kernel.shape
            # PT Convs are depthwise convs if and only if
            # `in_channels_per_group == 1`
            assert in_channels_per_group == 1
            pt_kernel = np.transpose(pt_kernel, (2, 3, 0, 1))
            in_channels = out_channels // depth_multiplier
            return np.reshape(
                pt_kernel, (height, width, in_channels, depth_multiplier)
            )

        loader.port_weight(
            keras_layer.kernel,
            hf_weight_key=f"{hf_weight_prefix}.weight",
            hook_fn=lambda x, _: convert_pt_conv2d_kernel(x),
        )

        if port_bias:
            loader.port_weight(
                keras_layer.bias,
                hf_weight_key=f"{hf_weight_prefix}.bias",
            )

    def port_batch_normalization(keras_layer, hf_weight_prefix):
        loader.port_weight(
            keras_layer.gamma,
            hf_weight_key=f"{hf_weight_prefix}.weight",
        )
        loader.port_weight(
            keras_layer.beta,
            hf_weight_key=f"{hf_weight_prefix}.bias",
        )
        loader.port_weight(
            keras_layer.moving_mean,
            hf_weight_key=f"{hf_weight_prefix}.running_mean",
        )
        loader.port_weight(
            keras_layer.moving_variance,
            hf_weight_key=f"{hf_weight_prefix}.running_var",
        )
        # do we need num batches tracked?

    # Stem
    port_conv2d(backbone.get_layer("stem_conv"), "conv_stem", port_bias=False)
    port_batch_normalization(backbone.get_layer("stem_bn"), "bn1")

    # Stages
    num_stacks = len(backbone.stackwise_kernel_sizes)

    for stack_index in range(num_stacks):
        block_type = backbone.stackwise_block_types[stack_index]
        expansion_ratio = backbone.stackwise_expansion_ratios[stack_index]
        repeats = backbone.stackwise_num_repeats[stack_index]
        stack_depth_coefficient = backbone.stackwise_depth_coefficients[
            stack_index
        ]

        repeats = int(math.ceil(stack_depth_coefficient * repeats))

        se_ratio = VARIANT_MAP[variant]["stackwise_squeeze_and_excite_ratios"][
            stack_index
        ]

        for block_idx in range(repeats):
            conv_pw_count = 0
            bn_count = 1

            # 97 is the start of the lowercase alphabet.
            letter_identifier = chr(block_idx + 97)

            keras_block_prefix = f"block{stack_index + 1}{letter_identifier}_"
            hf_block_prefix = f"blocks.{stack_index}.{block_idx}."

            if block_type == "v1":
                conv_pw_name_map = ["conv_pw", "conv_pwl"]
                # Initial Expansion Conv
                if expansion_ratio != 1:
                    port_conv2d(
                        backbone.get_layer(keras_block_prefix + "expand_conv"),
                        hf_block_prefix + conv_pw_name_map[conv_pw_count],
                        port_bias=False,
                    )
                    conv_pw_count += 1
                    port_batch_normalization(
                        backbone.get_layer(keras_block_prefix + "expand_bn"),
                        hf_block_prefix + f"bn{bn_count}",
                    )
                    bn_count += 1

                # Depthwise Conv
                port_depthwise_conv2d(
                    backbone.get_layer(keras_block_prefix + "dwconv"),
                    hf_block_prefix + "conv_dw",
                    port_bias=False,
                )
                port_batch_normalization(
                    backbone.get_layer(keras_block_prefix + "dwconv_bn"),
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1

                if 0 < se_ratio <= 1:
                    # Squeeze and Excite
                    port_conv2d(
                        backbone.get_layer(keras_block_prefix + "se_reduce"),
                        hf_block_prefix + "se.conv_reduce",
                    )
                    port_conv2d(
                        backbone.get_layer(keras_block_prefix + "se_expand"),
                        hf_block_prefix + "se.conv_expand",
                    )

                # Output/Projection
                port_conv2d(
                    backbone.get_layer(keras_block_prefix + "project"),
                    hf_block_prefix + conv_pw_name_map[conv_pw_count],
                    port_bias=False,
                )
                conv_pw_count += 1
                port_batch_normalization(
                    backbone.get_layer(keras_block_prefix + "project_bn"),
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1
            elif block_type == "fused":
                fused_block_layer = backbone.get_layer(keras_block_prefix)

                # Initial Expansion Conv
                port_conv2d(
                    fused_block_layer.conv1,
                    hf_block_prefix + "conv_exp",
                    port_bias=False,
                )
                conv_pw_count += 1
                port_batch_normalization(
                    fused_block_layer.bn1,
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1

                if 0 < se_ratio <= 1:
                    # Squeeze and Excite
                    port_conv2d(
                        fused_block_layer.se_conv1,
                        hf_block_prefix + "se.conv_reduce",
                    )
                    port_conv2d(
                        fused_block_layer.se_conv2,
                        hf_block_prefix + "se.conv_expand",
                    )

                # Output/Projection
                port_conv2d(
                    fused_block_layer.output_conv,
                    hf_block_prefix + "conv_pwl",
                    port_bias=False,
                )
                conv_pw_count += 1
                port_batch_normalization(
                    fused_block_layer.bn2,
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1

            elif block_type == "unfused":
                unfused_block_layer = backbone.get_layer(keras_block_prefix)
                # Initial Expansion Conv
                if expansion_ratio != 1:
                    port_conv2d(
                        unfused_block_layer.conv1,
                        hf_block_prefix + "conv_pw",
                        port_bias=False,
                    )
                    conv_pw_count += 1
                    port_batch_normalization(
                        unfused_block_layer.bn1,
                        hf_block_prefix + f"bn{bn_count}",
                    )
                    bn_count += 1

                # Depthwise Conv
                port_depthwise_conv2d(
                    unfused_block_layer.depthwise,
                    hf_block_prefix + "conv_dw",
                    port_bias=False,
                )
                port_batch_normalization(
                    unfused_block_layer.bn2,
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1

                if 0 < se_ratio <= 1:
                    # Squeeze and Excite
                    port_conv2d(
                        unfused_block_layer.se_conv1,
                        hf_block_prefix + "se.conv_reduce",
                    )
                    port_conv2d(
                        unfused_block_layer.se_conv2,
                        hf_block_prefix + "se.conv_expand",
                    )

                # Output/Projection
                port_conv2d(
                    unfused_block_layer.output_conv,
                    hf_block_prefix + "conv_pwl",
                    port_bias=False,
                )
                conv_pw_count += 1
                port_batch_normalization(
                    unfused_block_layer.bn3,
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1
            elif block_type == "cba":
                cba_block_layer = backbone.get_layer(keras_block_prefix)
                # Initial Expansion Conv
                port_conv2d(
                    cba_block_layer.conv1,
                    hf_block_prefix + "conv",
                    port_bias=False,
                )
                conv_pw_count += 1
                port_batch_normalization(
                    cba_block_layer.bn1,
                    hf_block_prefix + f"bn{bn_count}",
                )
                bn_count += 1

    # Head/Top
    port_conv2d(backbone.get_layer("top_conv"), "conv_head", port_bias=False)
    port_batch_normalization(backbone.get_layer("top_bn"), "bn2")


def convert_head(task, loader, timm_config):
    classifier_prefix = timm_config["pretrained_cfg"]["classifier"]
    prefix = f"{classifier_prefix}."
    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key=prefix + "weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key=prefix + "bias",
    )
