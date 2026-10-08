import types

import keras
import numpy as np

from keras_hub.src.models.mobilenetv5.mobilenetv5_attention import (
    MobileAttention,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import EdgeResidual
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import (
    UniversalInvertedResidual,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
    convert_arch_def_to_stackwise,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import RmsNorm2d

backbone_cls = MobileNetV5Backbone

MODEL_CONFIGS = {
    "mobilenetv5_300m": {
        "backbone": convert_arch_def_to_stackwise(
            [
                # Stage 0: 128x128 in
                [
                    "er_r1_k3_s2_e4_c128",
                    "er_r1_k3_s1_e4_c128",
                    "er_r1_k3_s1_e4_c128",
                ],
                # Stage 1: 256x256 in
                [
                    "uir_r1_a3_k5_s2_e6_c256",
                    "uir_r1_a5_k0_s1_e4_c256",
                    "uir_r1_a3_k0_s1_e4_c256",
                    "uir_r1_a5_k0_s1_e4_c256",
                    "uir_r1_a3_k0_s1_e4_c256",
                ],
                # Stage 2: 640x640 in
                [
                    "uir_r1_a5_k5_s2_e6_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a5_k0_s1_e4_c640",
                    "uir_r1_a0_k0_s1_e1_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                    "mqa_r1_k3_h12_v2_s1_d64_c640",
                    "uir_r1_a0_k0_s1_e2_c640",
                ],
                # Stage 3: 1280x1280 in
                [
                    "uir_r1_a5_k5_s2_e6_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                    "mqa_r1_k3_h16_s1_d96_c1280",
                    "uir_r1_a0_k0_s1_e2_c1280",
                ],
            ]
        ),
        "stem_size": 64,
        "num_features": 2048,
        "norm_layer": "rms_norm",
        "act_layer": "gelu",
        "use_msfa": True,
        "layer_scale_init_value": 1e-5,
    },
}


def convert_head(task, loader, timm_config):
    pass


def convert_backbone_config(timm_config):
    timm_architecture = timm_config["architecture"]
    if timm_architecture not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported architecture: {timm_architecture}")
    config = MODEL_CONFIGS[timm_architecture].copy()
    backbone_config = config.pop("backbone")
    backbone_config.update(config)
    return backbone_config


def convert_weights(backbone, loader, timm_config):
    def key_exists(key):
        try:
            loader.get_tensor(key)
            return True
        except Exception:
            return False

    def _port_weights(layer, timm_key, transpose_dims=None):
        hf_weight_key = f"{timm_key}.weight"
        if not key_exists(hf_weight_key):
            return
        hook_fn = None
        if transpose_dims:

            def transpose_hook(x, _):
                return np.transpose(x, transpose_dims)

            hook_fn = transpose_hook
        loader.port_weight(
            layer.kernel, hf_weight_key=hf_weight_key, hook_fn=hook_fn
        )
        if layer.bias is not None:
            hf_bias_key = f"{timm_key}.bias"
            if key_exists(hf_bias_key):
                loader.port_weight(
                    layer.bias,
                    hf_weight_key=hf_bias_key,
                )

    def _port_bn(layer, timm_prefix):
        loader.port_weight(layer.gamma, f"{timm_prefix}.weight")
        loader.port_weight(layer.beta, f"{timm_prefix}.bias")
        loader.port_weight(layer.moving_mean, f"{timm_prefix}.running_mean")
        loader.port_weight(layer.moving_variance, f"{timm_prefix}.running_var")

    def _port_rms_norm(layer, timm_prefix):
        loader.port_weight(layer.gamma, f"{timm_prefix}.weight")

    def _port_cna(cna_layer: ConvNormAct, timm_conv_prefix, timm_norm_prefix):
        if isinstance(cna_layer.conv, keras.layers.DepthwiseConv2D):
            _port_weights(
                cna_layer.conv,
                timm_conv_prefix,
                transpose_dims=(2, 3, 0, 1),
            )
        else:
            _port_weights(
                cna_layer.conv,
                timm_conv_prefix,
                transpose_dims=(2, 3, 1, 0),
            )
        if key_exists(f"{timm_norm_prefix}.running_mean"):
            _port_bn(cna_layer.norm, timm_norm_prefix)
        else:
            _port_rms_norm(cna_layer.norm, timm_norm_prefix)

    def _port_attn(attn_layer, attn_prefix):
        _port_weights(
            attn_layer.query_layers[-1],
            f"{attn_prefix}.query.proj",
            (2, 3, 1, 0),
        )
        if len(attn_layer.key_layers) > 1:
            _port_weights(
                attn_layer.key_layers[0],
                f"{attn_prefix}.key.down_conv",
                (2, 3, 0, 1),
            )
            key_norm_layer = attn_layer.key_layers[1]
            if isinstance(key_norm_layer, RmsNorm2d):
                _port_rms_norm(key_norm_layer, f"{attn_prefix}.key.norm")
            else:
                _port_bn(key_norm_layer, f"{attn_prefix}.key.norm")
        _port_weights(
            attn_layer.key_layers[-1], f"{attn_prefix}.key.proj", (2, 3, 1, 0)
        )
        if len(attn_layer.value_layers) > 1:
            _port_weights(
                attn_layer.value_layers[0],
                f"{attn_prefix}.value.down_conv",
                (2, 3, 0, 1),
            )
            value_norm_layer = attn_layer.value_layers[1]
            if isinstance(value_norm_layer, RmsNorm2d):
                _port_rms_norm(value_norm_layer, f"{attn_prefix}.value.norm")
            else:
                _port_bn(value_norm_layer, f"{attn_prefix}.value.norm")
        _port_weights(
            attn_layer.value_layers[-1],
            f"{attn_prefix}.value.proj",
            (2, 3, 1, 0),
        )
        _port_weights(
            attn_layer.output_proj_layers[-2],
            f"{attn_prefix}.output.proj",
            (2, 3, 1, 0),
        )

    stem_layer = backbone.get_layer("conv_stem")
    _port_cna(stem_layer, "conv_stem.conv", "conv_stem.bn")
    block_layers = [
        layer
        for layer in backbone.layers
        if isinstance(
            layer, (EdgeResidual, UniversalInvertedResidual, MobileAttention)
        )
    ]
    block_counter = 0
    for stack_idx in range(len(backbone.stackwise_num_blocks)):
        for block_idx_in_stage in range(
            backbone.stackwise_num_blocks[stack_idx]
        ):
            block = block_layers[block_counter]
            timm_prefix = f"blocks.{stack_idx}.{block_idx_in_stage}"
            if isinstance(block, EdgeResidual):
                _port_cna(
                    block.conv_exp,
                    f"{timm_prefix}.conv_exp",
                    f"{timm_prefix}.bn1",
                )
                _port_cna(
                    block.conv_pwl,
                    f"{timm_prefix}.conv_pwl",
                    f"{timm_prefix}.bn2",
                )
            elif isinstance(block, UniversalInvertedResidual):
                if hasattr(block, "dw_start") and not isinstance(
                    block.dw_start, types.FunctionType
                ):
                    _port_cna(
                        block.dw_start,
                        f"{timm_prefix}.dw_start.conv",
                        f"{timm_prefix}.dw_start.bn",
                    )
                _port_cna(
                    block.pw_exp,
                    f"{timm_prefix}.pw_exp.conv",
                    f"{timm_prefix}.pw_exp.bn",
                )
                if hasattr(block, "dw_mid") and not isinstance(
                    block.dw_mid, types.FunctionType
                ):
                    _port_cna(
                        block.dw_mid,
                        f"{timm_prefix}.dw_mid.conv",
                        f"{timm_prefix}.dw_mid.bn",
                    )
                _port_cna(
                    block.pw_proj,
                    f"{timm_prefix}.pw_proj.conv",
                    f"{timm_prefix}.pw_proj.bn",
                )
                gamma_key = f"{timm_prefix}.layer_scale.gamma"
                if key_exists(gamma_key):
                    loader.port_weight(block.layer_scale.gamma, gamma_key)
            elif isinstance(block, MobileAttention):
                _port_rms_norm(block.norm, f"{timm_prefix}.norm")
                gamma_key = f"{timm_prefix}.layer_scale.gamma"
                if key_exists(gamma_key):
                    loader.port_weight(block.layer_scale.gamma, gamma_key)
                attn_prefix = f"{timm_prefix}.attn"
                _port_attn(block.attn, attn_prefix)
            block_counter += 1
    try:
        msfa_layer = backbone.get_layer("msfa")
        ffn = msfa_layer.ffn
        _port_cna(ffn.pw_exp, "msfa.ffn.pw_exp.conv", "msfa.ffn.pw_exp.bn")
        _port_cna(ffn.pw_proj, "msfa.ffn.pw_proj.conv", "msfa.ffn.pw_proj.bn")
        _port_rms_norm(msfa_layer.norm, "msfa.norm")
    except ValueError:
        pass
