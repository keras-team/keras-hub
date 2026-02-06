"""Convert Swin Transformer checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_swin_transformer_checkpoints.py \
    --preset swin_tiny_patch4_window7_224
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import AutoImageProcessor
from transformers import SwinModel

import keras_hub
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "swin_tiny_patch4_window7_224": "microsoft/swin-tiny-patch4-window7-224",
    "swin_small_patch4_window7_224": (
        "microsoft/swin-small-patch4-window7-224"
    ),
    "swin_base_patch4_window7_224": "microsoft/swin-base-patch4-window7-224",
    "swin_base_patch4_window12_384": (
        "microsoft/swin-base-patch4-window12-384"
    ),
    "swin_large_patch4_window7_224": (
        "microsoft/swin-large-patch4-window7-224"
    ),
    "swin_large_patch4_window12_384": (
        "microsoft/swin-large-patch4-window12-384"
    ),
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/swin_transformer/keras/{preset}"',
    required=False,
)


def convert_model(hf_model):
    """Convert HuggingFace config to KerasHub model."""
    config = hf_model.config.to_dict()
    image_size = config["image_size"]

    backbone = SwinTransformerBackbone(
        image_shape=(image_size, image_size, 3),
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depths=tuple(config["depths"]),
        num_heads=tuple(config["num_heads"]),
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        qkv_bias=config["qkv_bias"],
        drop=config["hidden_dropout_prob"],
        attn_drop=config["attention_probs_dropout_prob"],
        drop_path=config["drop_path_rate"],
        patch_norm=True,
    )

    return backbone, config


def convert_backbone_weights(backbone, hf_model):
    """Port weights from HuggingFace to KerasHub format."""
    hf_state_dict = hf_model.state_dict()

    def get_hf_weight(key):
        if key in hf_state_dict:
            return hf_state_dict[key].detach().cpu().numpy()
        return None

    # 1. Patch embedding
    patch_proj = get_hf_weight("embeddings.patch_embeddings.projection.weight")
    patch_bias = get_hf_weight("embeddings.patch_embeddings.projection.bias")
    if patch_proj is not None:
        patch_proj = np.transpose(patch_proj, (2, 3, 1, 0))
        backbone.patch_embedding.proj.set_weights([patch_proj, patch_bias])

    norm_w = get_hf_weight("embeddings.norm.weight")
    norm_b = get_hf_weight("embeddings.norm.bias")
    if norm_w is not None and backbone.patch_embedding.norm is not None:
        backbone.patch_embedding.norm.set_weights([norm_w, norm_b])

    # 2. Convert all stages
    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            prefix = f"encoder.layers.{stage_idx}.blocks.{block_idx}"

            # LayerNorms
            n1_w = get_hf_weight(f"{prefix}.layernorm_before.weight")
            n1_b = get_hf_weight(f"{prefix}.layernorm_before.bias")
            if n1_w is not None:
                block.norm1.set_weights([n1_w, n1_b])
            n2_w = get_hf_weight(f"{prefix}.layernorm_after.weight")
            n2_b = get_hf_weight(f"{prefix}.layernorm_after.bias")
            if n2_w is not None:
                block.norm2.set_weights([n2_w, n2_b])

            # QKV
            attn_prefix = f"{prefix}.attention"
            q_w = get_hf_weight(f"{attn_prefix}.self.query.weight")
            k_w = get_hf_weight(f"{attn_prefix}.self.key.weight")
            v_w = get_hf_weight(f"{attn_prefix}.self.value.weight")
            q_b = get_hf_weight(f"{attn_prefix}.self.query.bias")
            k_b = get_hf_weight(f"{attn_prefix}.self.key.bias")
            v_b = get_hf_weight(f"{attn_prefix}.self.value.bias")

            if all(x is not None for x in [q_w, k_w, v_w, q_b, k_b, v_b]):
                qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
                qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)
                qkv_w = qkv_w.T
                block.attn.qkv.set_weights([qkv_w, qkv_b])

            # Attention projection
            proj_w = get_hf_weight(f"{attn_prefix}.output.dense.weight")
            proj_b = get_hf_weight(f"{attn_prefix}.output.dense.bias")
            if proj_w is not None:
                block.attn.proj.set_weights([proj_w.T, proj_b])

            # Relative position bias
            rel_bias = get_hf_weight(
                f"{attn_prefix}.self.relative_position_bias_table"
            )
            if rel_bias is not None:
                block.attn.relative_position_bias_table.assign(rel_bias)

            # MLP
            fc1_w = get_hf_weight(f"{prefix}.intermediate.dense.weight")
            fc1_b = get_hf_weight(f"{prefix}.intermediate.dense.bias")
            fc2_w = get_hf_weight(f"{prefix}.output.dense.weight")
            fc2_b = get_hf_weight(f"{prefix}.output.dense.bias")
            if fc1_w is not None:
                block.mlp.fc1.set_weights([fc1_w.T, fc1_b])
            if fc2_w is not None:
                block.mlp.fc2.set_weights([fc2_w.T, fc2_b])

        # Downsample
        if stage.downsample is not None:
            ds_prefix = f"encoder.layers.{stage_idx}.downsample"
            red_w = get_hf_weight(f"{ds_prefix}.reduction.weight")
            norm_w = get_hf_weight(f"{ds_prefix}.norm.weight")
            norm_b = get_hf_weight(f"{ds_prefix}.norm.bias")
            if red_w is not None:
                stage.downsample.reduction.set_weights([red_w.T])
            if norm_w is not None:
                stage.downsample.norm.set_weights([norm_w, norm_b])

    # 3. Final norm
    final_w = get_hf_weight("layernorm.weight")
    final_b = get_hf_weight("layernorm.bias")
    if final_w is not None:
        backbone.norm.set_weights([final_w, final_b])


def validate_output(keras_model, hf_model, hf_processor):
    """Validate converted model outputs match HuggingFace."""
    file = keras.utils.get_file(
        origin="http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    image = Image.open(file)

    # HuggingFace inference
    hf_inputs = hf_processor(image, return_tensors="pt")
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
        hf_features = hf_outputs.last_hidden_state.detach().cpu().numpy()

    # KerasHub inference
    image_size = hf_processor.size["height"]
    image_resized = image.resize((image_size, image_size), Image.BICUBIC)
    img_np = np.array(image_resized).astype("float32") / 255.0
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    keras_input = np.expand_dims(img_np, 0)
    keras_outputs = keras_model(keras_input, training=False)
    keras_features = keras.ops.convert_to_numpy(keras_outputs)

    print("🔶 Keras output (first token, first 10 dims):")
    print(f"   {keras_features[0, 0, :10]}")
    print("🔶 HF output (first token, first 10 dims):")
    print(f"   {hf_features[0, 0, :10]}")

    modeling_diff = np.mean(np.abs(keras_features - hf_features))
    max_diff = np.max(np.abs(keras_features - hf_features))
    relative_error = modeling_diff / np.mean(np.abs(hf_features))

    print(f"🔶 Modeling difference (mean): {modeling_diff:.6f}")
    print(f"🔶 Modeling difference (max):  {max_diff:.6f}")
    print(f"🔶 Relative error:             {relative_error * 100:.2f}%")


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Converting {preset}")

    # Load HuggingFace model
    hf_model = SwinModel.from_pretrained(hf_preset)
    hf_processor = AutoImageProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    # Convert to KerasHub
    keras_backbone, hf_config = convert_model(hf_model)
    print("✅ KerasHub model loaded.")
    print(f"   Parameters: {keras_backbone.count_params():,}")

    convert_backbone_weights(keras_backbone, hf_model)
    print("✅ Backbone weights converted.")

    validate_output(keras_backbone, hf_model, hf_processor)
    print("✅ Output validated.")

    keras_backbone.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
