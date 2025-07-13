import json
import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor
from transformers import DFineForObjectDetection

import keras_hub
from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_image_converter import (
    DFineImageConverter,
)
from keras_hub.src.models.d_fine.d_fine_layers import DFineConvNormLayer
from keras_hub.src.models.d_fine.d_fine_object_detector_preprocessor import (
    DFineObjectDetectorPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2ConvLayer
from keras_hub.src.models.hgnetv2.hgnetv2_layers import (
    HGNetV2LearnableAffineBlock,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Must be one of 'dfine_large_coco', 'dfine_xlarge_coco', "
    "'dfine_small_coco', 'dfine_nano_coco', 'dfine_medium_coco', "
    "'dfine_small_obj365', 'dfine_medium_obj365', 'dfine_large_obj365', "
    "'dfine_xlarge_obj365', 'dfine_small_obj2coco', 'dfine_medium_obj2coco', "
    "'dfine_large_obj2coco_e25', or 'dfine_xlarge_obj2coco'",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Optional upload URI, e.g., "kaggle://keras/dfine/keras/dfine_xlarge_coco"',
    required=False,
)

PRESET_MAP = {
    "dfine_large_coco": "ustc-community/dfine-large-coco",
    "dfine_xlarge_coco": "ustc-community/dfine-xlarge-coco",
    "dfine_small_coco": "ustc-community/dfine-small-coco",
    "dfine_medium_coco": "ustc-community/dfine-medium-coco",
    "dfine_nano_coco": "ustc-community/dfine-nano-coco",
    "dfine_small_obj365": "ustc-community/dfine-small-obj365",
    "dfine_medium_obj365": "ustc-community/dfine-medium-obj365",
    "dfine_large_obj365": "ustc-community/dfine-large-obj365",
    "dfine_xlarge_obj365": "ustc-community/dfine-xlarge-obj365",
    "dfine_small_obj2coco": "ustc-community/dfine-small-obj2coco",
    "dfine_medium_obj2coco": "ustc-community/dfine-medium-obj2coco",
    "dfine_large_obj2coco_e25": "ustc-community/dfine-large-obj2coco-e25",
    "dfine_xlarge_obj2coco": "ustc-community/dfine-xlarge-obj2coco",
}


def load_pytorch_model(hf_preset):
    model_path = hf_hub_download(
        repo_id=hf_preset,
        filename="model.safetensors",
        cache_dir="./hf_models",
    )
    state_dict = load_file(model_path)
    return state_dict


def get_keras_model(config):
    backbone_config = config["backbone_config"]
    stackwise_stage_filters = [
        [
            backbone_config["stage_in_channels"][i],
            backbone_config["stage_mid_channels"][i],
            backbone_config["stage_out_channels"][i],
            backbone_config["stage_num_blocks"][i],
            backbone_config["stage_numb_of_layers"][i],
            backbone_config["stage_kernel_size"][i],
        ]
        for i in range(len(backbone_config["stage_in_channels"]))
    ]
    hgnetv2_params = {
        "depths": backbone_config["depths"],
        "embedding_size": backbone_config["embedding_size"],
        "hidden_sizes": backbone_config["hidden_sizes"],
        "stem_channels": backbone_config["stem_channels"],
        "hidden_act": backbone_config["hidden_act"],
        "use_learnable_affine_block": backbone_config[
            "use_learnable_affine_block"
        ],
        "stackwise_stage_filters": stackwise_stage_filters,
        "apply_downsample": backbone_config["stage_downsample"],
        "use_lightweight_conv_block": backbone_config["stage_light_block"],
        "out_features": backbone_config["out_features"],
    }
    dfine_params = {
        "decoder_in_channels": config["decoder_in_channels"],
        "encoder_hidden_dim": config["encoder_hidden_dim"],
        "num_denoising": config["num_denoising"],
        "num_labels": len(config["id2label"]),
        "learn_initial_query": config["learn_initial_query"],
        "num_queries": config["num_queries"],
        "anchor_image_size": (640, 640),
        "feat_strides": config["feat_strides"],
        "batch_norm_eps": config["batch_norm_eps"],
        "num_feature_levels": config["num_feature_levels"],
        "hidden_dim": config["d_model"],
        "layer_norm_eps": config["layer_norm_eps"],
        "encoder_in_channels": config["encoder_in_channels"],
        "encode_proj_layers": config["encode_proj_layers"],
        "positional_encoding_temperature": config[
            "positional_encoding_temperature"
        ],
        "eval_size": config["eval_size"],
        "normalize_before": config["normalize_before"],
        "num_attention_heads": config["encoder_attention_heads"],
        "dropout": config["dropout"],
        "encoder_activation_function": config["encoder_activation_function"],
        "activation_dropout": config["activation_dropout"],
        "encoder_ffn_dim": config["encoder_ffn_dim"],
        "encoder_layers": config["encoder_layers"],
        "hidden_expansion": config["hidden_expansion"],
        "depth_mult": config["depth_mult"],
        "eval_idx": config["eval_idx"],
        "label_noise_ratio": config.get("label_noise_ratio", 0.5),
        "box_noise_scale": config.get("box_noise_scale", 1.0),
        "decoder_layers": config["decoder_layers"],
        "reg_scale": config["reg_scale"],
        "max_num_bins": config["max_num_bins"],
        "up": config.get("up", 0.5),
        "decoder_attention_heads": config["decoder_attention_heads"],
        "attention_dropout": config["attention_dropout"],
        "decoder_activation_function": config["decoder_activation_function"],
        "decoder_ffn_dim": config["decoder_ffn_dim"],
        "decoder_offset_scale": config["decoder_offset_scale"],
        "decoder_method": config["decoder_method"],
        "decoder_n_points": config["decoder_n_points"],
        "top_prob_values": config["top_prob_values"],
        "lqe_hidden_dim": config["lqe_hidden_dim"],
        "lqe_layers_count": config["lqe_layers"],
        "layer_scale": config.get("layer_scale", 1.0),
        "image_shape": (None, None, 3),
        "out_features": backbone_config["out_features"],
        "initializer_bias_prior_prob": config.get(
            "initializer_bias_prior_prob", None
        ),
        "initializer_range": config.get("initializer_range", 0.01),
        "seed": 0,
    }
    all_params = {**hgnetv2_params, **dfine_params}
    model = DFineBackbone(**all_params)
    return model


def set_conv_norm_weights(state_dict, prefix, k_conv):
    if isinstance(k_conv, HGNetV2ConvLayer):
        pt_conv_suffix = "convolution"
        pt_norm_suffix = "normalization"
        lab_suffix = "lab"
    elif isinstance(k_conv, DFineConvNormLayer):
        pt_conv_suffix = "conv"
        pt_norm_suffix = "norm"
        lab_suffix = None
    else:
        raise TypeError(f"Unsupported Keras ConvNormLayer type: {type(k_conv)}")
    conv_weight_key = f"{prefix}.{pt_conv_suffix}.weight"
    if conv_weight_key in state_dict:
        k_conv.convolution.kernel.assign(
            state_dict[conv_weight_key].permute(2, 3, 1, 0).numpy()
        )
    norm_weight_key = f"{prefix}.{pt_norm_suffix}.weight"
    norm_bias_key = f"{prefix}.{pt_norm_suffix}.bias"
    norm_mean_key = f"{prefix}.{pt_norm_suffix}.running_mean"
    norm_var_key = f"{prefix}.{pt_norm_suffix}.running_var"
    if all(
        key in state_dict
        for key in [norm_weight_key, norm_bias_key, norm_mean_key, norm_var_key]
    ):
        k_conv.normalization.set_weights(
            [
                state_dict[norm_weight_key].numpy(),
                state_dict[norm_bias_key].numpy(),
                state_dict[norm_mean_key].numpy(),
                state_dict[norm_var_key].numpy(),
            ]
        )
    if isinstance(k_conv, HGNetV2ConvLayer) and isinstance(
        k_conv.lab, HGNetV2LearnableAffineBlock
    ):
        lab_scale_key = f"{prefix}.{lab_suffix}.scale"
        lab_bias_key = f"{prefix}.{lab_suffix}.bias"
        if lab_scale_key in state_dict and lab_bias_key in state_dict:
            k_conv.lab.scale.assign(state_dict[lab_scale_key].item())
            k_conv.lab.bias.assign(state_dict[lab_bias_key].item())


def transfer_hgnet_backbone_weights(state_dict, k_backbone):
    backbone_prefix = "model.backbone.model."
    embedder_prefix = f"{backbone_prefix}embedder."
    for stem in ["stem1", "stem2a", "stem2b", "stem3", "stem4"]:
        k_conv = getattr(
            k_backbone.hgnetv2_backbone.embedder_layer, f"{stem}_layer"
        )
        set_conv_norm_weights(state_dict, f"{embedder_prefix}{stem}", k_conv)

    stages_prefix = f"{backbone_prefix}encoder.stages."
    for stage_idx, stage in enumerate(
        k_backbone.hgnetv2_backbone.encoder_layer.stages_list
    ):
        prefix = f"{stages_prefix}{stage_idx}."
        if hasattr(stage, "downsample_layer") and not isinstance(
            stage.downsample_layer, keras.layers.Identity
        ):
            set_conv_norm_weights(
                state_dict, f"{prefix}downsample", stage.downsample_layer
            )
        for block_idx, block in enumerate(stage.blocks_list):
            block_prefix = f"{prefix}blocks.{block_idx}."
            for layer_idx, layer in enumerate(block.layer_list):
                if hasattr(layer, "conv1_layer"):
                    set_conv_norm_weights(
                        state_dict,
                        f"{block_prefix}layers.{layer_idx}.conv1",
                        layer.conv1_layer,
                    )
                    set_conv_norm_weights(
                        state_dict,
                        f"{block_prefix}layers.{layer_idx}.conv2",
                        layer.conv2_layer,
                    )
                else:
                    set_conv_norm_weights(
                        state_dict, f"{block_prefix}layers.{layer_idx}", layer
                    )
            set_conv_norm_weights(
                state_dict,
                f"{block_prefix}aggregation.0",
                block.aggregation_squeeze_conv,
            )
            set_conv_norm_weights(
                state_dict,
                f"{block_prefix}aggregation.1",
                block.aggregation_excitation_conv,
            )


def transfer_hybrid_encoder_weights(state_dict, k_encoder):
    for i, lateral_conv in enumerate(k_encoder.lateral_convs_list):
        set_conv_norm_weights(
            state_dict, f"model.encoder.lateral_convs.{i}", lateral_conv
        )

    for i, fpn_block in enumerate(k_encoder.fpn_blocks_list):
        prefix = f"model.encoder.fpn_blocks.{i}"
        set_conv_norm_weights(state_dict, f"{prefix}.conv1", fpn_block.conv1)
        set_conv_norm_weights(state_dict, f"{prefix}.conv2", fpn_block.conv2)
        set_conv_norm_weights(state_dict, f"{prefix}.conv3", fpn_block.conv3)
        set_conv_norm_weights(state_dict, f"{prefix}.conv4", fpn_block.conv4)
        for j, bottleneck in enumerate(fpn_block.csp_rep1.bottleneck_layers):
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep1.bottlenecks.{j}.conv1",
                bottleneck.conv1_layer,
            )
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep1.bottlenecks.{j}.conv2",
                bottleneck.conv2_layer,
            )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep1.conv1", fpn_block.csp_rep1.conv1
        )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep1.conv2", fpn_block.csp_rep1.conv2
        )
        for j, bottleneck in enumerate(fpn_block.csp_rep2.bottleneck_layers):
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep2.bottlenecks.{j}.conv1",
                bottleneck.conv1_layer,
            )
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep2.bottlenecks.{j}.conv2",
                bottleneck.conv2_layer,
            )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep2.conv1", fpn_block.csp_rep2.conv1
        )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep2.conv2", fpn_block.csp_rep2.conv2
        )

    for i, down_conv in enumerate(k_encoder.downsample_convs_list):
        prefix = f"model.encoder.downsample_convs.{i}"
        set_conv_norm_weights(state_dict, f"{prefix}.conv1", down_conv.conv1)
        set_conv_norm_weights(state_dict, f"{prefix}.conv2", down_conv.conv2)

    for i, pan_block in enumerate(k_encoder.pan_blocks_list):
        prefix = f"model.encoder.pan_blocks.{i}"
        set_conv_norm_weights(state_dict, f"{prefix}.conv1", pan_block.conv1)
        set_conv_norm_weights(state_dict, f"{prefix}.conv2", pan_block.conv2)
        set_conv_norm_weights(state_dict, f"{prefix}.conv3", pan_block.conv3)
        set_conv_norm_weights(state_dict, f"{prefix}.conv4", pan_block.conv4)
        for j, bottleneck in enumerate(pan_block.csp_rep1.bottleneck_layers):
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep1.bottlenecks.{j}.conv1",
                bottleneck.conv1_layer,
            )
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep1.bottlenecks.{j}.conv2",
                bottleneck.conv2_layer,
            )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep1.conv1", pan_block.csp_rep1.conv1
        )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep1.conv2", pan_block.csp_rep1.conv2
        )
        for j, bottleneck in enumerate(pan_block.csp_rep2.bottleneck_layers):
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep2.bottlenecks.{j}.conv1",
                bottleneck.conv1_layer,
            )
            set_conv_norm_weights(
                state_dict,
                f"{prefix}.csp_rep2.bottlenecks.{j}.conv2",
                bottleneck.conv2_layer,
            )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep2.conv1", pan_block.csp_rep2.conv1
        )
        set_conv_norm_weights(
            state_dict, f"{prefix}.csp_rep2.conv2", pan_block.csp_rep2.conv2
        )


def transfer_transformer_encoder_weights(state_dict, k_encoder):
    for i, layer in enumerate(k_encoder.encoder_list[0].encoder_layer_list):
        prefix = f"model.encoder.encoder.0.layers.{i}"
        for proj in ["q", "k", "v"]:
            pt_weight = state_dict[
                f"{prefix}.self_attn.{proj}_proj.weight"
            ].T.numpy()
            head_dim = (
                k_encoder.encoder_hidden_dim // k_encoder.num_attention_heads
            )
            k_weight = pt_weight.reshape(
                k_encoder.encoder_hidden_dim,
                k_encoder.num_attention_heads,
                head_dim,
            )
            k_proj = getattr(layer.self_attn, f"{proj}_proj")
            k_proj.weights[0].assign(k_weight)
            k_proj.weights[1].assign(
                state_dict[f"{prefix}.self_attn.{proj}_proj.bias"]
                .numpy()
                .reshape(k_encoder.num_attention_heads, head_dim)
            )
        layer.self_attn.out_proj.weights[0].assign(
            state_dict[f"{prefix}.self_attn.out_proj.weight"].T.numpy()
        )
        layer.self_attn.out_proj.weights[1].assign(
            state_dict[f"{prefix}.self_attn.out_proj.bias"].numpy()
        )
        layer.self_attn_layer_norm.set_weights(
            [
                state_dict[f"{prefix}.self_attn_layer_norm.weight"].numpy(),
                state_dict[f"{prefix}.self_attn_layer_norm.bias"].numpy(),
            ]
        )
        layer.fc1.weights[0].assign(
            state_dict[f"{prefix}.fc1.weight"].T.numpy()
        )
        layer.fc1.weights[1].assign(state_dict[f"{prefix}.fc1.bias"].numpy())
        layer.fc2.weights[0].assign(
            state_dict[f"{prefix}.fc2.weight"].T.numpy()
        )
        layer.fc2.weights[1].assign(state_dict[f"{prefix}.fc2.bias"].numpy())
        layer.final_layer_norm.set_weights(
            [
                state_dict[f"{prefix}.final_layer_norm.weight"].numpy(),
                state_dict[f"{prefix}.final_layer_norm.bias"].numpy(),
            ]
        )


def transfer_decoder_weights(state_dict, k_decoder):
    for i, layer in enumerate(k_decoder.decoder_layers):
        prefix = f"model.decoder.layers.{i}"
        for proj in ["q", "k", "v"]:
            pt_weight = state_dict[
                f"{prefix}.self_attn.{proj}_proj.weight"
            ].T.numpy()
            head_dim = k_decoder.hidden_dim // k_decoder.decoder_attention_heads
            k_weight = pt_weight.reshape(
                k_decoder.hidden_dim,
                k_decoder.decoder_attention_heads,
                head_dim,
            )
            k_proj = getattr(layer.self_attn, f"{proj}_proj")
            k_proj.weights[0].assign(k_weight)
            k_proj.weights[1].assign(
                state_dict[f"{prefix}.self_attn.{proj}_proj.bias"]
                .numpy()
                .reshape(k_decoder.decoder_attention_heads, head_dim)
            )
        layer.self_attn.out_proj.weights[0].assign(
            state_dict[f"{prefix}.self_attn.out_proj.weight"].T.numpy()
        )
        layer.self_attn.out_proj.weights[1].assign(
            state_dict[f"{prefix}.self_attn.out_proj.bias"].numpy()
        )
        layer.self_attn_layer_norm.set_weights(
            [
                state_dict[f"{prefix}.self_attn_layer_norm.weight"].numpy(),
                state_dict[f"{prefix}.self_attn_layer_norm.bias"].numpy(),
            ]
        )
        layer.encoder_attn.sampling_offsets.weights[0].assign(
            state_dict[
                f"{prefix}.encoder_attn.sampling_offsets.weight"
            ].T.numpy()
        )
        layer.encoder_attn.sampling_offsets.weights[1].assign(
            state_dict[f"{prefix}.encoder_attn.sampling_offsets.bias"].numpy()
        )
        layer.encoder_attn.attention_weights.weights[0].assign(
            state_dict[
                f"{prefix}.encoder_attn.attention_weights.weight"
            ].T.numpy()
        )
        layer.encoder_attn.attention_weights.weights[1].assign(
            state_dict[f"{prefix}.encoder_attn.attention_weights.bias"].numpy()
        )
        num_points_scale_key = f"{prefix}.encoder_attn.num_points_scale"
        if num_points_scale_key in state_dict:
            layer.encoder_attn.num_points_scale.assign(
                state_dict[num_points_scale_key].numpy()
            )
        layer.fc1.weights[0].assign(
            state_dict[f"{prefix}.fc1.weight"].T.numpy()
        )
        layer.fc1.weights[1].assign(state_dict[f"{prefix}.fc1.bias"].numpy())
        layer.fc2.weights[0].assign(
            state_dict[f"{prefix}.fc2.weight"].T.numpy()
        )
        layer.fc2.weights[1].assign(state_dict[f"{prefix}.fc2.bias"].numpy())
        layer.final_layer_norm.set_weights(
            [
                state_dict[f"{prefix}.final_layer_norm.weight"].numpy(),
                state_dict[f"{prefix}.final_layer_norm.bias"].numpy(),
            ]
        )
        layer.gateway.gate.weights[0].assign(
            state_dict[f"{prefix}.gateway.gate.weight"].T.numpy()
        )
        layer.gateway.gate.weights[1].assign(
            state_dict[f"{prefix}.gateway.gate.bias"].numpy()
        )
        layer.gateway.norm.set_weights(
            [
                state_dict[f"{prefix}.gateway.norm.weight"].numpy(),
                state_dict[f"{prefix}.gateway.norm.bias"].numpy(),
            ]
        )

    for i, layer in enumerate(k_decoder.lqe_layers):
        prefix = f"model.decoder.lqe_layers.{i}.reg_conf.layers"
        for j, dense in enumerate(layer.reg_conf.dense_layers):
            dense.weights[0].assign(
                state_dict[f"{prefix}.{j}.weight"].T.numpy()
            )
            dense.weights[1].assign(state_dict[f"{prefix}.{j}.bias"].numpy())

    for i, dense in enumerate(k_decoder.pre_bbox_head.dense_layers):
        prefix = f"model.decoder.pre_bbox_head.layers.{i}"
        dense.weights[0].assign(state_dict[f"{prefix}.weight"].T.numpy())
        dense.weights[1].assign(state_dict[f"{prefix}.bias"].numpy())

    for i, dense in enumerate(k_decoder.query_pos_head.dense_layers):
        prefix = f"model.decoder.query_pos_head.layers.{i}"
        dense.weights[0].assign(state_dict[f"{prefix}.weight"].T.numpy())
        dense.weights[1].assign(state_dict[f"{prefix}.bias"].numpy())

    k_decoder.reg_scale.assign(state_dict["model.decoder.reg_scale"].numpy())
    k_decoder.up.assign(state_dict["model.decoder.up"].numpy())


def transfer_prediction_heads(state_dict, k_decoder):
    for i, class_embed in enumerate(k_decoder.class_embed):
        prefix = f"model.decoder.class_embed.{i}"
        class_embed.weights[0].assign(state_dict[f"{prefix}.weight"].T.numpy())
        class_embed.weights[1].assign(state_dict[f"{prefix}.bias"].numpy())
    for i, bbox_embed in enumerate(k_decoder.bbox_embed):
        prefix = f"model.decoder.bbox_embed.{i}.layers"
        for j, layer in enumerate(bbox_embed.dense_layers):
            layer.weights[0].assign(
                state_dict[f"{prefix}.{j}.weight"].T.numpy()
            )
            layer.weights[1].assign(state_dict[f"{prefix}.{j}.bias"].numpy())


def transfer_dfine_model_weights(state_dict, backbone):
    transfer_hgnet_backbone_weights(state_dict, backbone)

    for i, proj_seq in enumerate(backbone.encoder_input_proj):
        prefix = f"model.encoder_input_proj.{i}"
        conv_weight_key = f"{prefix}.0.weight"
        if conv_weight_key in state_dict:
            proj_seq.layers[0].weights[0].assign(
                state_dict[conv_weight_key].permute(2, 3, 1, 0).numpy()
            )
            proj_seq.layers[1].set_weights(
                [
                    state_dict[f"{prefix}.1.weight"].numpy(),
                    state_dict[f"{prefix}.1.bias"].numpy(),
                    state_dict[f"{prefix}.1.running_mean"].numpy(),
                    state_dict[f"{prefix}.1.running_var"].numpy(),
                ]
            )

    transfer_hybrid_encoder_weights(state_dict, backbone.encoder)
    transfer_transformer_encoder_weights(state_dict, backbone.encoder)
    if backbone.denoising_class_embed is not None:
        backbone.denoising_class_embed.weights[0].assign(
            state_dict["model.denoising_class_embed.weight"].numpy()
        )

    backbone.enc_output.layers[0].weights[0].assign(
        state_dict["model.enc_output.0.weight"].T.numpy()
    )
    backbone.enc_output.layers[0].weights[1].assign(
        state_dict["model.enc_output.0.bias"].numpy()
    )
    backbone.enc_output.layers[1].set_weights(
        [
            state_dict["model.enc_output.1.weight"].numpy(),
            state_dict["model.enc_output.1.bias"].numpy(),
        ]
    )

    backbone.enc_score_head.weights[0].assign(
        state_dict["model.enc_score_head.weight"].T.numpy()
    )
    backbone.enc_score_head.weights[1].assign(
        state_dict["model.enc_score_head.bias"].numpy()
    )

    for i, dense in enumerate(backbone.enc_bbox_head.dense_layers):
        prefix = f"model.enc_bbox_head.layers.{i}"
        dense.weights[0].assign(state_dict[f"{prefix}.weight"].T.numpy())
        dense.weights[1].assign(state_dict[f"{prefix}.bias"].numpy())

    for i, proj_seq in enumerate(backbone.decoder_input_proj):
        prefix = f"model.decoder_input_proj.{i}"
        if isinstance(proj_seq, keras.layers.Identity):
            continue
        conv_weight_key = f"{prefix}.0.weight"
        if conv_weight_key in state_dict:
            proj_seq.layers[0].weights[0].assign(
                state_dict[conv_weight_key].permute(2, 3, 1, 0).numpy()
            )
            proj_seq.layers[1].set_weights(
                [
                    state_dict[f"{prefix}.1.weight"].numpy(),
                    state_dict[f"{prefix}.1.bias"].numpy(),
                    state_dict[f"{prefix}.1.running_mean"].numpy(),
                    state_dict[f"{prefix}.1.running_var"].numpy(),
                ]
            )

    transfer_decoder_weights(state_dict, backbone.decoder)
    transfer_prediction_heads(state_dict, backbone.decoder)


def validate_conversion(keras_model, hf_preset):
    pt_model = DFineForObjectDetection.from_pretrained(hf_preset)
    image_processor = AutoImageProcessor.from_pretrained(hf_preset)
    pt_model.eval()
    raw_image = np.random.uniform(0, 255, (640, 640, 3)).astype(np.uint8)
    pil_image = Image.fromarray(raw_image)
    inputs = image_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        pt_outputs = pt_model(**inputs)
    config_path = keras.utils.get_file(
        origin=f"https://huggingface.co/{hf_preset}/raw/main/preprocessor_config.json",  # noqa: E501
        cache_subdir=f"hf_models/{hf_preset}",
    )
    with open(config_path, "r") as f:
        preprocessor_config = json.load(f)
    scale = None
    offset = None
    if preprocessor_config.get("do_rescale", False):
        scale = preprocessor_config.get("rescale_factor")
    if preprocessor_config.get("do_normalize", False):
        mean = preprocessor_config["image_mean"]
        std = preprocessor_config["image_std"]
        if isinstance(scale, (float, int)):
            scale = [scale / s for s in std]
        else:
            scale = [1.0 / s for s in std]
        offset = [-m / s for m, s in zip(mean, std)]
    image_converter = DFineImageConverter(
        image_size=(640, 640),
        scale=scale,
        offset=offset,
        crop_to_aspect_ratio=True,
    )
    preprocessor = DFineObjectDetectorPreprocessor(
        image_converter=image_converter,
    )
    keras_input = np.expand_dims(raw_image, axis=0).astype(np.float32)
    keras_preprocessed_input = preprocessor(keras_input)
    keras_outputs = keras_model(keras_preprocessed_input, training=False)
    intermediate_logits = keras_outputs["intermediate_logits"]
    k_logits = intermediate_logits[:, -1, :, :]
    k_pred_boxes = keras_outputs["intermediate_reference_points"][:, -1, :, :]

    def to_numpy(tensor):
        if keras.backend.backend() == "torch":
            return tensor.detach().numpy()
        elif keras.backend.backend() == "jax":
            return np.array(tensor)
        elif keras.backend.backend() == "tensorflow":
            return tensor.numpy()
        else:
            return np.array(tensor)

    pt_pred_boxes = pt_outputs["pred_boxes"].detach().cpu().numpy()
    print("\n=== Output Comparison ===")
    pt_logits = pt_outputs["logits"].detach().cpu().numpy()
    k_logits = to_numpy(k_logits)
    k_pred_boxes = to_numpy(k_pred_boxes)
    boxes_diff = np.mean(np.abs(pt_pred_boxes - k_pred_boxes))
    if boxes_diff < 1e-5:
        print(f"🔶 Predicted Bounding Boxes Difference: {boxes_diff:.6e}")
        print("✅ Validation successful")
    print(f"PyTorch Logits Shape: {pt_logits.shape}, dtype: {pt_logits.dtype}")
    print(f"Keras Logits Shape: {k_logits.shape}, dtype: {k_logits.dtype}")
    print("\n=== Logits Statistics ===")
    print(
        f"PyTorch Logits Min: {np.min(pt_logits):.6f}, max: "
        f"{np.max(pt_logits):.6f}, mean: {np.mean(pt_logits):.6f}, std: "
        f"{np.std(pt_logits):.6f}"
    )
    print(
        f"Keras Logits Min: {np.min(k_logits):.6f}, max: {np.max(k_logits):.6f}"
        f", mean: {np.mean(k_logits):.6f}, std: {np.std(k_logits):.6f}"
    )
    print("\n=== Pred Boxes Statistics ===")
    print(
        f"PyTorch Pred Boxes Min: {np.min(pt_pred_boxes):.6f}, max: "
        f"{np.max(pt_pred_boxes):.6f}, mean: {np.mean(pt_pred_boxes):.6f}, "
        f"std: {np.std(pt_pred_boxes):.6f}"
    )
    print(
        f"Keras Pred Boxes Min: {np.min(k_pred_boxes):.6f}, max: "
        f"{np.max(k_pred_boxes):.6f}, mean: {np.mean(k_pred_boxes):.6f}, std: "
        f"{np.std(k_pred_boxes):.6f}"
    )
    print(f"NaN in Keras Logits: {np.any(np.isnan(k_logits))}")
    print(f"NaN in Keras Boxes: {np.any(np.isnan(k_pred_boxes))}")


def main(_):
    keras.utils.set_random_seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one of "
            f"{list(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[FLAGS.preset]
    output_dir = FLAGS.preset
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"\n✅ Converting {FLAGS.preset}")

    state_dict = load_pytorch_model(hf_preset)
    print("✅ PyTorch state dict loaded")

    config_path = hf_hub_download(
        repo_id=hf_preset,
        filename="config.json",
        cache_dir="./hf_models",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    keras_model = get_keras_model(config)
    dummy_input = np.zeros((1, 640, 640, 3), dtype="float32")
    keras_model(dummy_input)
    print("✅ Keras model constructed")

    transfer_dfine_model_weights(state_dict, keras_model)
    print("✅ Weights transferred")
    validate_conversion(keras_model, hf_preset)
    print("✅ Validation completed")

    keras_model.save_to_preset(output_dir)
    print(f"🏁 Preset saved to {output_dir}")

    if FLAGS.upload_uri:
        keras_hub.upload_preset(uri=FLAGS.upload_uri, preset=output_dir)
        print(f"🏁 Preset uploaded to {FLAGS.upload_uri}")


if __name__ == "__main__":
    app.run(main)
