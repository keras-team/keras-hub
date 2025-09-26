import os
import shutil
import types

import keras
import numpy as np
import PIL
import timm
import torch
from absl import app
from absl import flags

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
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import decode_arch_def
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier import (
    MobileNetV5ImageClassifier,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier_preprocessor import (  # noqa: E501
    MobileNetV5ImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_converter import (
    MobileNetV5ImageConverter,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import RmsNorm2d

PRESET_MAP = {
    "mobilenetv5_300m_enc.gemma3n": {
        "arch": "mobilenetv5_300m_enc",
        "hf_hub_id": "timm/mobilenetv5_300m.gemma3n",
    }
}

MODEL_CONFIGS = {
    "mobilenetv5_300m_enc.gemma3n": {
        "backbone": {
            "block_args": decode_arch_def(
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
        "classifier": {
            "num_classes": 0,
        },
    }
}


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `MobileNetV5` preset.",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Optional Kaggle URI to upload the converted model preset.",
    required=False,
)


class TimmToKerasConverter:
    def __init__(self, timm_model):
        self.state_dict = {
            k: v.cpu().numpy() for k, v in timm_model.state_dict().items()
        }

    def convert(self, keras_model: MobileNetV5ImageClassifier):
        print("🔶 Starting weight conversion...")
        backbone = keras_model.backbone
        self._port_stem(backbone)
        self._port_blocks(backbone)
        self._port_msfa(backbone)
        print("✅ Backbone weights converted.")

    def _port_weights(self, layer, timm_key, transpose_dims=None):
        if timm_key not in self.state_dict:
            print(f"⚠️ Weight key not found in state_dict: {timm_key}")
            return
        weights = self.state_dict[timm_key]
        if transpose_dims:
            weights = weights.transpose(transpose_dims)

        current_weights = layer.get_weights()
        if len(current_weights) == 1:
            layer.set_weights([weights])
        elif len(current_weights) == 2:
            bias_key = timm_key.replace(".weight", ".bias")
            if bias_key in self.state_dict:
                bias = self.state_dict[bias_key]
                layer.set_weights([weights, bias])
            else:
                layer.set_weights([weights, current_weights[1]])
        else:
            print(f"❓ Unexpected number of weights in layer {layer.name}")

    def _port_bn(self, layer, timm_prefix):
        keys = [
            f"{timm_prefix}.weight",
            f"{timm_prefix}.bias",
            f"{timm_prefix}.running_mean",
            f"{timm_prefix}.running_var",
        ]
        weights = [self.state_dict[key] for key in keys]
        layer.set_weights(weights)

    def _port_rms_norm(self, layer, timm_prefix):
        key = f"{timm_prefix}.weight"
        layer.set_weights([self.state_dict[key]])

    def _port_cna(
        self, cna_layer: ConvNormAct, timm_conv_prefix, timm_norm_prefix
    ):
        if isinstance(cna_layer.conv, keras.layers.DepthwiseConv2D):
            self._port_weights(
                cna_layer.conv,
                f"{timm_conv_prefix}.weight",
                transpose_dims=(2, 3, 0, 1),
            )
        else:
            self._port_weights(
                cna_layer.conv,
                f"{timm_conv_prefix}.weight",
                transpose_dims=(2, 3, 1, 0),
            )
        if f"{timm_norm_prefix}.running_mean" in self.state_dict:
            self._port_bn(cna_layer.norm, timm_norm_prefix)
        else:
            self._port_rms_norm(cna_layer.norm, timm_norm_prefix)

    def _port_stem(self, backbone):
        print("  -> Porting stem...")
        stem_layer = backbone.get_layer("conv_stem")
        self._port_cna(stem_layer, "conv_stem.conv", "conv_stem.bn")

    def _port_msfa(self, backbone):
        print("  -> Porting MSFA...")
        try:
            msfa_layer = backbone.get_layer("msfa")
            ffn = msfa_layer.ffn
            self._port_cna(
                ffn.pw_exp, "msfa.ffn.pw_exp.conv", "msfa.ffn.pw_exp.bn"
            )
            self._port_cna(
                ffn.pw_proj, "msfa.ffn.pw_proj.conv", "msfa.ffn.pw_proj.bn"
            )
            self._port_rms_norm(msfa_layer.norm, "msfa.norm")
        except ValueError:
            print("  -> MSFA layer not found, skipping.")

    def _port_blocks(self, backbone: MobileNetV5Backbone):
        print("  -> Porting blocks...")
        block_layers = [
            layer
            for layer in backbone.layers
            if isinstance(
                layer,
                (EdgeResidual, UniversalInvertedResidual, MobileAttention),
            )
        ]
        block_counter = 0
        for stack_idx, stack_args in enumerate(backbone.block_args):
            print(f"    -> Stack {stack_idx}")
            for block_idx_in_stage in range(len(stack_args)):
                block = block_layers[block_counter]
                timm_prefix = f"blocks.{stack_idx}.{block_idx_in_stage}"
                if isinstance(block, EdgeResidual):
                    self._port_cna(
                        block.conv_exp,
                        f"{timm_prefix}.conv_exp",
                        f"{timm_prefix}.bn1",
                    )
                    self._port_cna(
                        block.conv_pwl,
                        f"{timm_prefix}.conv_pwl",
                        f"{timm_prefix}.bn2",
                    )
                elif isinstance(block, UniversalInvertedResidual):
                    if hasattr(block, "dw_start") and not isinstance(
                        block.dw_start, types.FunctionType
                    ):
                        self._port_cna(
                            block.dw_start,
                            f"{timm_prefix}.dw_start.conv",
                            f"{timm_prefix}.dw_start.bn",
                        )
                    self._port_cna(
                        block.pw_exp,
                        f"{timm_prefix}.pw_exp.conv",
                        f"{timm_prefix}.pw_exp.bn",
                    )
                    if hasattr(block, "dw_mid") and not isinstance(
                        block.dw_mid, types.FunctionType
                    ):
                        self._port_cna(
                            block.dw_mid,
                            f"{timm_prefix}.dw_mid.conv",
                            f"{timm_prefix}.dw_mid.bn",
                        )
                    self._port_cna(
                        block.pw_proj,
                        f"{timm_prefix}.pw_proj.conv",
                        f"{timm_prefix}.pw_proj.bn",
                    )
                    gamma_key = f"{timm_prefix}.layer_scale.gamma"
                    if gamma_key in self.state_dict:
                        block.layer_scale.set_weights(
                            [self.state_dict[gamma_key]]
                        )
                elif isinstance(block, MobileAttention):
                    self._port_rms_norm(block.norm, f"{timm_prefix}.norm")
                    gamma_key = f"{timm_prefix}.layer_scale.gamma"
                    if gamma_key in self.state_dict:
                        block.layer_scale.set_weights(
                            [self.state_dict[gamma_key]]
                        )
                    attn_prefix = f"{timm_prefix}.attn"
                    self._port_attn(block.attn, attn_prefix)
                block_counter += 1

    def _port_attn(self, attn_layer, attn_prefix):
        self._port_weights(
            attn_layer.query_layers[-1],
            f"{attn_prefix}.query.proj.weight",
            (2, 3, 1, 0),
        )
        if len(attn_layer.key_layers) > 1:
            self._port_weights(
                attn_layer.key_layers[0],
                f"{attn_prefix}.key.down_conv.weight",
                (2, 3, 0, 1),
            )
            key_norm_layer = attn_layer.key_layers[1]
            if isinstance(key_norm_layer, RmsNorm2d):
                self._port_rms_norm(key_norm_layer, f"{attn_prefix}.key.norm")
            else:
                self._port_bn(key_norm_layer, f"{attn_prefix}.key.norm")
        self._port_weights(
            attn_layer.key_layers[-1],
            f"{attn_prefix}.key.proj.weight",
            (2, 3, 1, 0),
        )
        if len(attn_layer.value_layers) > 1:
            self._port_weights(
                attn_layer.value_layers[0],
                f"{attn_prefix}.value.down_conv.weight",
                (2, 3, 0, 1),
            )
            value_norm_layer = attn_layer.value_layers[1]
            if isinstance(value_norm_layer, RmsNorm2d):
                self._port_rms_norm(
                    value_norm_layer, f"{attn_prefix}.value.norm"
                )
            else:
                self._port_bn(value_norm_layer, f"{attn_prefix}.value.norm")
        self._port_weights(
            attn_layer.value_layers[-1],
            f"{attn_prefix}.value.proj.weight",
            (2, 3, 1, 0),
        )
        self._port_weights(
            attn_layer.output_proj_layers[-2],
            f"{attn_prefix}.output.proj.weight",
            (2, 3, 1, 0),
        )


def validate_output(keras_model, timm_model):
    file = keras.utils.get_file(
        origin=(
            "https://storage.googleapis.com/keras-cv/"
            "models/paligemma/cow_beach_1.png"
        )
    )
    image = PIL.Image.open(file)
    batch = np.array([image])

    # Preprocess with Timm.
    data_config = timm.data.resolve_model_data_config(timm_model)
    data_config["crop_pct"] = 1.0  # Stop timm from cropping.
    transforms = timm.data.create_transform(**data_config, is_training=False)
    timm_preprocessed = transforms(image)
    timm_preprocessed = keras.ops.transpose(timm_preprocessed, axes=(1, 2, 0))
    timm_preprocessed = keras.ops.expand_dims(timm_preprocessed, 0)

    # Preprocess with Keras.
    batch = keras.ops.cast(batch, "float32")
    keras_preprocessed = keras_model.preprocessor(batch)

    # Call with Timm. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    timm_batch = keras.ops.transpose(keras_preprocessed, axes=(0, 3, 1, 2))
    timm_batch = torch.from_numpy(np.array(timm_batch))
    timm_outputs = timm_model(timm_batch).detach().numpy()

    # Call with Keras.
    keras_outputs = keras_model.predict(batch)
    keras_label = np.argmax(keras_outputs[0])

    # Apply global average pooling to the Timm output to match Keras's output
    timm_outputs_pooled = np.mean(timm_outputs, axis=(2, 3))

    print("🔶 Keras output:", keras_outputs[0, :10])
    print("🔶 TIMM output (pooled):", timm_outputs_pooled[0, :10])
    print("🔶 Keras label:", keras_label)
    modeling_diff = np.mean(np.abs(keras_outputs - timm_outputs_pooled))
    print("🔶 Modeling difference:", modeling_diff)


def main(_):
    preset = FLAGS.preset
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)
    timm_config = PRESET_MAP[preset]
    timm_arch = timm_config["arch"]
    hf_hub_id = timm_config["hf_hub_id"]
    print(f"✅ Loading TIMM model: {timm_arch} from {hf_hub_id}")
    timm_model = timm.create_model(
        timm_arch,
        pretrained=True,
        pretrained_cfg_overlay=dict(hf_hub_id=hf_hub_id),
    )
    timm_model = timm_model.eval()
    print("✅ Creating Keras model.")
    config = MODEL_CONFIGS[preset]
    backbone = MobileNetV5Backbone(**config["backbone"])
    pretrained_cfg = timm_model.pretrained_cfg
    image_size = (
        pretrained_cfg["input_size"][1],
        pretrained_cfg["input_size"][2],
    )
    mean = pretrained_cfg["mean"]
    std = pretrained_cfg["std"]
    interpolation = pretrained_cfg["interpolation"]
    scale = [1 / (255.0 * s) for s in std] if std else 1 / 255.0
    offset = [-m / s for m, s in zip(mean, std)] if mean and std else 0.0
    image_converter = MobileNetV5ImageConverter(
        image_size=image_size,
        scale=scale,
        offset=offset,
        interpolation=interpolation,
        antialias=True if interpolation == "bicubic" else False,
    )
    preprocessor = MobileNetV5ImageClassifierPreprocessor(
        image_converter=image_converter
    )
    keras_model = MobileNetV5ImageClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        **config["classifier"],
    )
    converter = TimmToKerasConverter(timm_model)
    converter.convert(keras_model)
    validate_output(keras_model, timm_model)
    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}")
    upload_uri = FLAGS.upload_uri
    if upload_uri:
        try:
            import keras_hub

            keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
            print(f"🏁 Preset uploaded to {upload_uri}")
        except ImportError:
            print("❗ `keras-hub` is not installed. Skipping upload.")
        except Exception as e:
            print(f"❗ An error occurred during upload: {e}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
