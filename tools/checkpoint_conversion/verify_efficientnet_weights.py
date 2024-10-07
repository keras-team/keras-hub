# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Verify EfficientNet weight checkpoints.

python tools/checkpoint_conversion/verify_efficientnet_weights.py \
    --preset enet_b0_ra 
"""

import os
import shutil

import keras
import numpy as np
import PIL
import timm
import torch
from absl import app
from absl import flags

import keras_hub

PRESET_MAP = {
    "enet_b0_ra": "timm/efficientnet_b0.ra_in1k",
    "enet_b1_ft": "timm/efficientnet_b1.ft_in1k",
    "enet_b1_pruned": "timm/efficientnet_b1_pruned.in1k",
    "enet_b2_ra": "timm/efficientnet_b2.ra_in1k",
    "enet_b2_pruned": "timm/efficientnet_b2_pruned.in1k",
    "enet_b3_ra2": "timm/efficientnet_b3.ra2_in1k",
    "enet_b3_pruned": "timm/efficientnet_b3_pruned.in1k",
    "enet_b4_ra2": "timm/efficientnet_b4.ra2_in1k",
    "enet_b5_sw": "timm/efficientnet_b5.sw_in12k",
    "enet_b5_sw_ft": "timm/efficientnet_b5.sw_in12k_ft_in1k",
    "enet_el_ra": "timm/efficientnet_el.ra_in1k",
    "enet_el_pruned": "timm/efficientnet_el_pruned.in1k",
    "enet_em_ra2": "timm/efficientnet_em.ra2_in1k",
    "enet_es_ra": "timm/efficientnet_es.ra_in1k",
    "enet_es_pruned": "timm/efficientnet_es_pruned.in1k",
    "enet_b0_ra4_e3600_r224": "timm/efficientnet_b0.ra4_e3600_r224_in1k",
    "enet_b1_ra4_e3600_r240": "timm/efficientnet_b1.ra4_e3600_r240_in1k",
    "enet2_rw_m_agc": "timm/efficientnetv2_rw_m.agc_in1k",
    "enet2_rw_s_ra2": "timm/efficientnetv2_rw_s.ra2_in1k",
    "enet2_rw_t_ra2": "timm/efficientnetv2_rw_t.ra2_in1k",
}
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `EfficientNet` preset from KerasHub",
    required=True,
)


def construct_op_map(keras_model):
    backbone = keras_model.backbone

    op_map = {
        "conv_stem": {
            "keras_name": "stem_conv",
            "type": "Conv2D",
            "include_bias": False,
        },
        "bn1": {
            "keras_name": "stem_bn",
            "type": "BatchNormalization",
        }
    }

    num_stacks = len(backbone.stackwise_kernel_sizes)
    for stack_index in range(num_stacks):
        block_type = backbone.stackwise_block_types[stack_index]
        expansion_ratio = backbone.stackwise_expansion_ratios[stack_index]

        for block_idx in range(backbone.stackwise_num_repeats[stack_index]):

            conv_pw_count = 0
            bn_count = 1
            conv_pw_name_map = ["conv_pw", "conv_pwl"]

            # 97 is the start of the lowercase alphabet.
            letter_identifier = chr(block_idx + 97)

            if block_type == "v1":
                keras_block_prefix = f"block{stack_index+1}{letter_identifier}_"
                hf_block_prefix = f"blocks.{stack_index}.{block_idx}."

                # Initial Expansion Conv
                if expansion_ratio != 1:
                    op_map[hf_block_prefix + conv_pw_name_map[conv_pw_count]] = {
                        "keras_name": keras_block_prefix + "expand_conv",
                        "type": "Conv2D",
                        "include_bias": False,
                    }
                    conv_pw_count += 1
                    op_map[hf_block_prefix + f"bn{bn_count}"] = {
                        "keras_name": keras_block_prefix + "expand_bn",
                        "type": "BatchNormalization",
                    }
                    bn_count += 1

                # Depthwise Conv
                op_map[hf_block_prefix + "conv_dw"] = {
                    "keras_name": keras_block_prefix + "dwconv",
                    "type": "DepthwiseConv2D",
                    "include_bias": False,
                }
                op_map[hf_block_prefix + f"bn{bn_count}"] = {
                    "keras_name": keras_block_prefix + "dwconv_bn",
                    "type": "BatchNormalization",
                }
                bn_count += 1
                
                # Squeeze and Excite
                op_map[hf_block_prefix + "se.conv_reduce"] = {
                    "keras_name": keras_block_prefix + "se_reduce",
                    "type": "Conv2D",
                    "include_bias": True,
                }
                op_map[hf_block_prefix + "se.conv_expand"] = {
                    "keras_name": keras_block_prefix + "se_expand",
                    "type": "Conv2D",
                    "include_bias": True,
                }
                
                # Output/Projection
                op_map[hf_block_prefix + conv_pw_name_map[conv_pw_count]] = {
                    "keras_name": keras_block_prefix + "project",
                    "type": "Conv2D",
                    "include_bias": False,
                }
                conv_pw_count += 1
                op_map[hf_block_prefix + f"bn{bn_count}"] = {
                    "keras_name": keras_block_prefix + "project_bn",
                    "type": "BatchNormalization",
                }
                bn_count += 1

    op_map["conv_head"] = {
        "keras_name": "top_conv",
        "type": "Conv2D",
        "include_bias": False,
    }
    op_map["bn2"] = {
        "keras_name": "top_bn",
        "type": "BatchNormalization",
    }

    return op_map


def verify_weights(keras_model, timm_model):
    op_map = construct_op_map(keras_model)

    timm_state_dict = timm_model.state_dict()
    keras_backbone = keras_model.backbone

    for timm_name, op_data in op_map.items():
        keras_layer = keras_backbone.get_layer(op_data["keras_name"])

        if op_data["type"] == "Conv2D":
            include_bias = op_data["include_bias"]
            kernel_tensor = keras_layer.get_weights()[0]
            bias_tensor = keras_layer.get_weights()[1] if include_bias else None

            timm_kernel_tensor = timm_state_dict[timm_name + ".weight"]
            timm_bias_tensor = timm_state_dict[timm_name + ".bias"] if include_bias else None

            assert np.allclose(kernel_tensor, np.transpose(timm_kernel_tensor, (2, 3, 1, 0)), atol=1e-6)
            if include_bias:
                assert np.allclose(bias_tensor, timm_bias_tensor, atol=1e-6)
        elif op_data["type"] == "DepthwiseConv2D":
            include_bias = op_data["include_bias"]
            kernel_tensor = keras_layer.get_weights()[0]
            bias_tensor = keras_layer.get_weights()[1] if include_bias else None

            timm_kernel_tensor = timm_state_dict[timm_name + ".weight"]
            timm_bias_tensor = timm_state_dict[timm_name + ".bias"] if include_bias else None

            assert np.allclose(kernel_tensor, np.transpose(timm_kernel_tensor, (2, 3, 0, 1)), atol=1e-6)
            if include_bias:
                assert np.allclose(bias_tensor, timm_bias_tensor, atol=1e-6)
        elif op_data["type"] == "BatchNormalization":
            gamma, beta, moving_mean, moving_variance = keras_layer.get_weights()

            timm_gamma = timm_state_dict[timm_name + ".weight"]
            timm_beta = timm_state_dict[timm_name + ".bias"]
            timm_moving_mean = timm_state_dict[timm_name + ".running_mean"]
            timm_moving_variance = timm_state_dict[timm_name + ".running_var"]

            assert np.allclose(gamma, timm_gamma, atol=1e-6)
            assert np.allclose(beta, timm_beta, atol=1e-6)
            assert np.allclose(moving_mean, timm_moving_mean, atol=1e-6)
            assert np.allclose(moving_variance, timm_moving_variance, atol=1e-6)
        else:
            raise ValueError(f"Unknown layer type: {op_data['type']}")
            
    print("✅ All weights match.")


def main(_):
    preset = FLAGS.preset
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    timm_name = PRESET_MAP[preset]

    print("✅ Loaded TIMM model.")
    timm_model = timm.create_model(timm_name, pretrained=True)
    timm_model = timm_model.eval()

    print("✅ Loaded KerasHub model.")
    keras_model = keras_hub.models.ImageClassifier.from_preset(
        "hf://" + timm_name,
    )

    verify_weights(keras_model, timm_model)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
