"""Convert HGNetV2 checkpoints from Hugging Face to Keras.

Usage:
    export KAGGLE_USERNAME=XXX
    export KAGGLE_KEY=XXX

    python tools/checkpoint_conversion/convert_hgnetv2_checkpoints.py
"""

import json
import os
import shutil

import keras
import numpy as np
import safetensors.torch
import torch
from absl import app
from absl import flags
from PIL import Image
from timm import create_model

import keras_hub
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier import (
    HGNetV2ImageClassifier,
)
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier_preprocessor import (
    HGNetV2ImageClassifierPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_image_converter import (
    HGNetV2ImageConverter,
)
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2ConvLayerLight
from keras_hub.src.models.hgnetv2.hgnetv2_layers import (
    HGNetV2LearnableAffineBlock,
)
from keras_hub.src.utils.imagenet.imagenet_utils import (
    decode_imagenet_predictions,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "hgnetv2_b6_ssld_stage2_ft_in1k": "timm/hgnetv2_b6.ssld_stage2_ft_in1k",
    "hgnetv2_b6_ssld_stage1_in22k_in1k": "timm/hgnetv2_b6.ssld_stage1_in22k_in1k",  # noqa: E501
    "hgnetv2_b5_ssld_stage2_ft_in1k": "timm/hgnetv2_b5.ssld_stage2_ft_in1k",
    "hgnetv2_b5_ssld_stage1_in22k_in1k": "timm/hgnetv2_b5.ssld_stage1_in22k_in1k",  # noqa: E501
    "hgnetv2_b4_ssld_stage2_ft_in1k": "timm/hgnetv2_b4.ssld_stage2_ft_in1k",
}
LAB_FALSE_PRESETS = [
    "hgnetv2_b6_ssld_stage2_ft_in1k",
    "hgnetv2_b6_ssld_stage1_in22k_in1k",
    "hgnetv2_b5_ssld_stage2_ft_in1k",
    "hgnetv2_b5_ssld_stage1_in22k_in1k",
    "hgnetv2_b4_ssld_stage2_ft_in1k",
]

flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/hgnetv2/keras/{preset}"',
    required=False,
)
HGNETV2_CONFIGS = {
    "hgnetv2_b0": {
        "stem_channels": [3, 16, 16],
        "stackwise_stage_filters": [
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 16,
        "hidden_sizes": [64, 256, 512, 1024],
        "depths": [1, 1, 2, 1],
    },
    "hgnetv2_b1": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 64, 1, 3, 3],
            [64, 48, 256, 1, 3, 3],
            [256, 96, 512, 2, 3, 5],
            [512, 192, 1024, 1, 3, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [64, 256, 512, 1024],
        "depths": [1, 1, 2, 1],
    },
    "hgnetv2_b2": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 96, 1, 4, 3],
            [96, 64, 384, 1, 4, 3],
            [384, 128, 768, 3, 4, 5],
            [768, 256, 1536, 1, 4, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [96, 384, 768, 1536],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b3": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 128, 1, 5, 3],
            [128, 64, 512, 1, 5, 3],
            [512, 128, 1024, 3, 5, 5],
            [1024, 256, 2048, 1, 5, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b4": {
        "stem_channels": [3, 32, 48],
        "stackwise_stage_filters": [
            [48, 48, 128, 1, 6, 3],
            [128, 96, 512, 1, 6, 3],
            [512, 192, 1024, 3, 6, 5],
            [1024, 384, 2048, 1, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 48,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b5": {
        "stem_channels": [3, 32, 64],
        "stackwise_stage_filters": [
            [64, 64, 128, 1, 6, 3],
            [128, 128, 512, 2, 6, 3],
            [512, 256, 1024, 5, 6, 5],
            [1024, 512, 2048, 2, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 64,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 2, 5, 2],
    },
    "hgnetv2_b6": {
        "stem_channels": [3, 48, 96],
        "stackwise_stage_filters": [
            [96, 96, 192, 2, 6, 3],
            [192, 192, 512, 3, 6, 3],
            [512, 384, 1024, 6, 6, 5],
            [1024, 768, 2048, 3, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 96,
        "hidden_sizes": [192, 512, 1024, 2048],
        "depths": [2, 3, 6, 3],
    },
}


def load_hf_config(hf_preset, hf_model):
    config_path = keras.utils.get_file(
        origin=f"https://huggingface.co/{hf_preset}/raw/main/config.json",
        cache_subdir=f"hf_models/{hf_preset}",
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    config["pretrained_cfg"] = hf_model.pretrained_cfg
    return config


def convert_model(hf_config, architecture, preset_name):
    config = HGNETV2_CONFIGS[architecture]
    image_size = hf_config["pretrained_cfg"]["input_size"][1]
    use_lab = preset_name not in LAB_FALSE_PRESETS

    backbone = HGNetV2Backbone(
        image_shape=(image_size, image_size, 3),
        depths=config["depths"],
        embedding_size=config["embedding_size"],
        hidden_sizes=config["hidden_sizes"],
        stem_channels=config["stem_channels"],
        hidden_act="relu",
        use_learnable_affine_block=use_lab,
        stackwise_stage_filters=config["stackwise_stage_filters"],
        apply_downsample=config["apply_downsample"],
        use_lightweight_conv_block=config["use_lightweight_conv_block"],
    )
    pretrained_cfg = hf_config["pretrained_cfg"]
    image_size = (
        pretrained_cfg["input_size"][1],
        pretrained_cfg["input_size"][2],
    )
    mean = pretrained_cfg["mean"]
    std = pretrained_cfg["std"]
    interpolation = pretrained_cfg["interpolation"]
    scale = [1 / (255.0 * s) for s in std] if std else 1 / 255.0
    offset = [-m / s for m, s in zip(mean, std)] if mean and std else 0
    image_converter = HGNetV2ImageConverter(
        image_size=image_size,
        scale=scale,
        offset=offset,
        interpolation=interpolation,
        antialias=True if interpolation == "bicubic" else False,
    )
    preprocessor = HGNetV2ImageClassifierPreprocessor(
        image_converter=image_converter
    )
    keras_model = HGNetV2ImageClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=hf_config["num_classes"],
    )
    return keras_model, config, image_size


def convert_weights(keras_model, hf_model):
    state_dict = hf_model.state_dict()

    def port_weights(keras_variable, weight_key, hook_fn=None):
        if weight_key not in state_dict:
            raise KeyError(f"Weight key '{weight_key}' not found in state_dict")
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        if (
            keras_variable.shape == ()
            and isinstance(torch_tensor, np.ndarray)
            and torch_tensor.shape == (1,)
        ):
            torch_tensor = torch_tensor[0]
        keras_variable.assign(torch_tensor)

    def port_last_conv(keras_conv_layer, state_dict, prefix="head.last_conv"):
        port_weights(
            keras_conv_layer.convolution.kernel,
            f"{prefix}.0.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        if f"{prefix}.1.weight" in state_dict:
            port_weights(
                keras_conv_layer.normalization.gamma, f"{prefix}.1.weight"
            )
            port_weights(
                keras_conv_layer.normalization.beta, f"{prefix}.1.bias"
            )
            port_weights(
                keras_conv_layer.normalization.moving_mean,
                f"{prefix}.1.running_mean",
            )
            port_weights(
                keras_conv_layer.normalization.moving_variance,
                f"{prefix}.1.running_var",
            )
            if isinstance(keras_conv_layer.lab, HGNetV2LearnableAffineBlock):
                lab_scale_key = f"{prefix}.2.scale"
                lab_bias_key = f"{prefix}.2.bias"
                if lab_scale_key in state_dict:
                    port_weights(keras_conv_layer.lab.scale, lab_scale_key)
                    port_weights(keras_conv_layer.lab.bias, lab_bias_key)
        else:
            gamma_dtype = keras_conv_layer.normalization.gamma.dtype
            if isinstance(gamma_dtype, keras.DTypePolicy):
                gamma_dtype = gamma_dtype.name
            gamma_identity_value = np.sqrt(
                1.0 + keras_conv_layer.normalization.epsilon
            ).astype(gamma_dtype)
            keras_conv_layer.normalization.gamma.assign(
                np.full_like(
                    keras_conv_layer.normalization.gamma.numpy(),
                    gamma_identity_value,
                )
            )
            keras_conv_layer.normalization.beta.assign(
                np.zeros_like(keras_conv_layer.normalization.beta.numpy())
            )
            keras_conv_layer.normalization.moving_mean.assign(
                np.zeros_like(
                    keras_conv_layer.normalization.moving_mean.numpy()
                )
            )
            keras_conv_layer.normalization.moving_variance.assign(
                np.ones_like(
                    keras_conv_layer.normalization.moving_variance.numpy()
                )
            )
            if isinstance(keras_conv_layer.lab, HGNetV2LearnableAffineBlock):
                lab_scale_key_idx1 = f"{prefix}.1.scale"
                lab_bias_key_idx1 = f"{prefix}.1.bias"
                lab_scale_key_idx2 = f"{prefix}.2.scale"
                lab_bias_key_idx2 = f"{prefix}.2.bias"
                if lab_scale_key_idx1 in state_dict:
                    port_weights(keras_conv_layer.lab.scale, lab_scale_key_idx1)
                    port_weights(keras_conv_layer.lab.bias, lab_bias_key_idx1)
                elif lab_scale_key_idx2 in state_dict:
                    port_weights(keras_conv_layer.lab.scale, lab_scale_key_idx2)
                    port_weights(keras_conv_layer.lab.bias, lab_bias_key_idx2)

    def port_conv(keras_conv_layer, weight_key_prefix):
        port_weights(
            keras_conv_layer.convolution.kernel,
            f"{weight_key_prefix}.conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        port_weights(
            keras_conv_layer.normalization.gamma,
            f"{weight_key_prefix}.bn.weight",
        )
        port_weights(
            keras_conv_layer.normalization.beta,
            f"{weight_key_prefix}.bn.bias",
        )
        port_weights(
            keras_conv_layer.normalization.moving_mean,
            f"{weight_key_prefix}.bn.running_mean",
        )
        port_weights(
            keras_conv_layer.normalization.moving_variance,
            f"{weight_key_prefix}.bn.running_var",
        )
        if not isinstance(keras_conv_layer.lab, keras.layers.Identity):
            port_weights(
                keras_conv_layer.lab.scale, f"{weight_key_prefix}.lab.scale"
            )
            port_weights(
                keras_conv_layer.lab.bias, f"{weight_key_prefix}.lab.bias"
            )

    def port_conv_light(keras_conv_light_layer, weight_key_prefix):
        port_conv(
            keras_conv_light_layer.conv1_layer, f"{weight_key_prefix}.conv1"
        )
        port_conv(
            keras_conv_light_layer.conv2_layer, f"{weight_key_prefix}.conv2"
        )

    def port_embeddings(keras_embeddings, weight_key_prefix):
        port_conv(keras_embeddings.stem1_layer, f"{weight_key_prefix}.stem1")
        port_conv(keras_embeddings.stem2a_layer, f"{weight_key_prefix}.stem2a")
        port_conv(keras_embeddings.stem2b_layer, f"{weight_key_prefix}.stem2b")
        port_conv(keras_embeddings.stem3_layer, f"{weight_key_prefix}.stem3")
        port_conv(keras_embeddings.stem4_layer, f"{weight_key_prefix}.stem4")

    def port_basic_layer(keras_basic_layer, weight_key_prefix):
        for i, layer in enumerate(keras_basic_layer.layer_list):
            layer_prefix = f"{weight_key_prefix}.layers.{i}"
            if isinstance(layer, HGNetV2ConvLayerLight):
                port_conv_light(layer, layer_prefix)
            else:
                port_conv(layer, layer_prefix)
        port_conv(
            keras_basic_layer.aggregation_squeeze_conv,
            f"{weight_key_prefix}.aggregation.0",
        )
        port_conv(
            keras_basic_layer.aggregation_excitation_conv,
            f"{weight_key_prefix}.aggregation.1",
        )

    def port_stage(keras_stage, weight_key_prefix):
        if not isinstance(keras_stage.downsample_layer, keras.layers.Identity):
            port_conv(
                keras_stage.downsample_layer, f"{weight_key_prefix}.downsample"
            )
        for block_idx, block in enumerate(keras_stage.blocks_list):
            port_basic_layer(block, f"{weight_key_prefix}.blocks.{block_idx}")

    def port_encoder(keras_encoder, weight_key_prefix):
        for i, stage in enumerate(keras_encoder.stages_list):
            port_stage(stage, f"{weight_key_prefix}.{i}")

    port_embeddings(keras_model.backbone.embedder_layer, "stem")
    port_encoder(keras_model.backbone.encoder_layer, "stages")
    port_last_conv(keras_model.last_conv, state_dict)
    port_weights(
        keras_model.output_dense.kernel,
        "head.fc.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )
    port_weights(keras_model.output_dense.bias, "head.fc.bias")


def convert_image_converter(hf_config):
    pretrained_cfg = hf_config["pretrained_cfg"]
    image_size = (
        pretrained_cfg["input_size"][1],
        pretrained_cfg["input_size"][2],
    )
    mean = pretrained_cfg["mean"]
    std = pretrained_cfg["std"]
    interpolation = pretrained_cfg["interpolation"]
    scale = [1 / (255.0 * s) for s in std] if std else 1 / 255.0
    offset = [-m / s for m, s in zip(mean, std)] if mean and std else 0
    image_converter = HGNetV2ImageConverter(
        image_size=image_size,
        scale=scale,
        offset=offset,
        interpolation=interpolation,
        antialias=True if interpolation == "bicubic" else False,
    )
    return image_converter, mean, std


def validate_output(keras_model, keras_image_converter, hf_model, mean, std):
    file = keras.utils.get_file(
        origin="https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg"
    )
    image = Image.open(file)
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    keras_preprocessed = keras_model.preprocessor(images)
    keras_preprocessed = keras.ops.convert_to_tensor(
        keras_preprocessed, dtype="float32"
    )
    keras_logits = keras_model.predict(keras_preprocessed)
    hf_inputs = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )
    keras_backbone_output_dict = keras_model.backbone(
        keras_preprocessed, training=False
    )
    last_stage_name = keras_model.backbone.stage_names[-1]
    keras_last_stage_tensor = keras_backbone_output_dict[last_stage_name]
    hf_backbone = torch.nn.Sequential(*list(hf_model.children())[:-1])
    hf_backbone_output = hf_backbone(hf_inputs)
    keras_output_np = keras.ops.convert_to_numpy(keras_last_stage_tensor)
    hf_output_np = hf_backbone_output.detach().cpu().numpy()
    hf_output_np = np.transpose(hf_output_np, (0, 2, 3, 1))
    print(
        "🔶 Modeling Difference (Mean Absolute):",
        np.mean(np.abs(keras_output_np - hf_output_np)),
    )
    print(
        "🔬 Keras top 5 ImageNet predictions:",
        decode_imagenet_predictions(keras_logits, top=5),
    )


def main(_):
    for preset in PRESET_MAP.keys():
        hf_preset = PRESET_MAP[preset]
        if os.path.exists(preset):
            shutil.rmtree(preset)
        os.makedirs(preset)

        print(f"\n🏃 Converting {preset}")
        hf_model = create_model(hf_preset, pretrained=True)
        safetensors_file = keras.utils.get_file(
            origin=f"https://huggingface.co/{hf_preset}/resolve/main/model.safetensors",
            cache_subdir=f"hf_models/{hf_preset}",
        )
        try:
            state_dict = safetensors.torch.load_file(safetensors_file)
            hf_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading Safetensors file for {preset}: {e}")
            print("Clearing cache and retrying download...")
            cache_dir = os.path.dirname(safetensors_file)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            safetensors_file = keras.utils.get_file(
                origin=f"https://huggingface.co/{hf_preset}/resolve/main/model.safetensors",
                cache_subdir=f"hf_models/{hf_preset}",
            )
            state_dict = safetensors.torch.load_file(safetensors_file)
        hf_model.eval()
        hf_config = load_hf_config(hf_preset, hf_model)
        architecture = hf_config["architecture"]
        keras_model, _, _ = convert_model(hf_config, architecture, preset)
        print("✅ KerasHub model loaded.")
        convert_weights(keras_model, hf_model)
        print("✅ Weights converted.")
        keras_image_converter, mean, std = convert_image_converter(hf_config)
        validate_output(keras_model, keras_image_converter, hf_model, mean, std)
        print("✅ Output validated.")
        keras_model.save_to_preset(f"./{preset}")
        print(f"🏁 Preset saved to ./{preset}.")
        upload_uri = FLAGS.upload_uri
        if upload_uri:
            keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
            print(f"🏁 Successfully uploaded {preset} to {upload_uri}")
    print("\n🏁 All presets validated and saved successfully!")


if __name__ == "__main__":
    app.run(main)
