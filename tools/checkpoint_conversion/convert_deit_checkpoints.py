"""Convert DeiT checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_deit_checkpoints.py \
    --preset deit-base-distilled-patch16-384
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import DeiTForImageClassificationWithTeacher
from transformers import DeiTImageProcessor

import keras_hub
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_image_classifier import DeiTImageClassifier
from keras_hub.src.models.deit.deit_image_classifier_preprocessor import (
    DeiTImageClassifierPreprocessor,
)
from keras_hub.src.models.deit.deit_image_converter import DeiTImageConverter

FLAGS = flags.FLAGS

PRESET_MAP = {
    "deit_base_distilled_patch16_384_imagenet": (
        "facebook/deit-base-distilled-patch16-384"
    ),
    "deit_base_distilled_patch16_224_imagenet": (
        "facebook/deit-base-distilled-patch16-224"
    ),
    "deit_small_distilled_patch16_224_imagenet": (
        "facebook/deit-small-distilled-patch16-224"
    ),
    "deit_tiny_distilled_patch16_224_imagenet": (
        "facebook/deit-tiny-distilled-patch16-224"
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
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def convert_model(hf_model):
    config = hf_model.config.to_dict()
    image_size = config["image_size"]
    backbone = DeiTBackbone(
        image_shape=(image_size, image_size, 3),
        patch_size=config["patch_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["hidden_size"],
        intermediate_dim=config["intermediate_size"],
        dropout_rate=config["hidden_dropout_prob"],
        attention_dropout=config["attention_probs_dropout_prob"],
        layer_norm_epsilon=config["layer_norm_eps"],
        # TODO as hf equivalent is not found yet
        # use_mha_bias=config["qkv_bias"],
    )

    return backbone, config


def convert_backbone_weights(backbone, hf_model):
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())

    # Helper functions.
    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_ln(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_mha(keras_variable, weight_key, num_heads, hidden_dim):
        # query
        port_weights(
            keras_variable.query_dense.kernel,
            f"{weight_key}.attention.query.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.attention.query.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.attention.key.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.attention.key.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.attention.value.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.attention.value.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.output.dense.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.output.dense.bias"
        )

    port_weights(
        keras_variable=backbone.layers[1].patch_embedding.kernel,
        weight_key="deit.embeddings.patch_embeddings.projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    port_weights(
        backbone.layers[1].patch_embedding.bias,
        "deit.embeddings.patch_embeddings.projection.bias",
    )

    port_weights(
        backbone.layers[1].class_token,
        "deit.embeddings.cls_token",
    )

    port_weights(
        backbone.layers[1].distillation_token,
        "deit.embeddings.distillation_token",
    )

    port_weights(
        backbone.layers[1].position_embedding,
        "deit.embeddings.position_embeddings",
    )

    encoder_layers = backbone.layers[2].encoder_layers
    for i, encoder_block in enumerate(encoder_layers):
        prefix = "deit.encoder.layer"
        num_heads = encoder_block.num_heads
        hidden_dim = encoder_block.hidden_dim

        port_mha(
            encoder_block.mha,
            f"{prefix}.{i}.attention",
            num_heads,
            hidden_dim,
        )
        port_ln(encoder_block.layer_norm_1, f"{prefix}.{i}.layernorm_before")
        port_ln(encoder_block.layer_norm_2, f"{prefix}.{i}.layernorm_after")

        port_dense(encoder_block.mlp.dense, f"{prefix}.{i}.intermediate.dense")
        port_dense(
            encoder_block.output_layer.dense, f"{prefix}.{i}.output.dense"
        )
    port_ln(backbone.layers[2].layer_norm, "deit.layernorm")


def convert_head_weights(keras_model, hf_model):
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())

    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    prefix = "cls_classifier."
    port_weights(
        keras_model.output_dense.kernel,
        weight_key=prefix + "weight",
        hook_fn=lambda x, _: x.T,
    )
    port_weights(
        keras_model.output_dense.bias,
        weight_key=prefix + "bias",
    )


def convert_image_converter(hf_image_processor):
    config = hf_image_processor.to_dict()
    # Huggingface converter does center_crop after resizing to convert to
    # required image size, so crop_size is the image_size provided to model
    image_size = (config["crop_size"]["height"], config["crop_size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    scale = [config["rescale_factor"] / s for s in std]
    return DeiTImageConverter(
        image_size=image_size,
        scale=scale,
        offset=[-m / s for m, s in zip(mean, std)],
        antialias=True,  # True for matching with hf preset
        interpolation="bicubic",  # DeiT defaults to bicubic resampling.
        crop_to_aspect_ratio=False,  # for matching outputs with hf preprocessor
    )


def validate_output(
    keras_model,
    keras_image_converter,
    hf_model,
    hf_image_processor,
    head_weights=True,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)

    # Compare number of parameters between Keras and HF backbone
    keras_params = keras_model.backbone.count_params()
    hf_params = sum(p.numel() for n, p in hf_model.deit.named_parameters())

    print(f"🔶 Keras model params: {keras_params:,}")
    print(f"🔶 HF model params:    {hf_params:,}")
    assert keras_params == hf_params, (
        "❌ Parameter count mismatch between Keras and HF models!"
    )

    # Preprocess with hf.
    hf_inputs = hf_image_processor(
        image,
        return_tensors="pt",
    )
    hf_preprocessed = hf_inputs["pixel_values"].detach().cpu().numpy()

    # Preprocess with keras.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    images = np.concatenate([images, images], axis=0)
    images = keras_image_converter(images)
    keras_preprocessed = keras.ops.convert_to_numpy(images)

    # Call with hf. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )

    hf_outputs = hf_model(**hf_inputs)
    if head_weights:
        hf_vision_logits = hf_outputs.cls_logits.detach().cpu().numpy()

    else:
        # When validating the backbone only (without classification head)
        hf_vision_logits = hf_outputs.last_hidden_state.detach().cpu().numpy()

    # Call with keras.
    keras_outputs = keras_model(keras_preprocessed)
    keras_vision_logits = keras.ops.convert_to_numpy(keras_outputs)

    print("🔶 Keras output:", keras_vision_logits[0, :5])
    print("🔶 HF output:", hf_vision_logits[0, :5])
    if head_weights:
        print(
            "🔶 HF top 5 ImageNet outputs:",
            keras_hub.utils.decode_imagenet_predictions(hf_vision_logits),
        )
        print(
            "🔶 Keras top 5 ImageNet outputs:",
            keras_hub.utils.decode_imagenet_predictions(keras_outputs),
        )
    modeling_diff = np.mean(np.abs(keras_vision_logits - hf_vision_logits))
    print("🔶 Modeling difference:", modeling_diff)
    preprocessing_diff = np.mean(
        np.abs(keras_preprocessed - np.transpose(hf_preprocessed, (0, 2, 3, 1)))
    )
    print("🔶 Preprocessing difference:", preprocessing_diff)


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

    print(f"🏃 Coverting {preset}")

    # Load huggingface model.
    hf_model = DeiTForImageClassificationWithTeacher.from_pretrained(hf_preset)
    # Load preprocessor
    hf_preprocessor = DeiTImageProcessor.from_pretrained(
        hf_preset,
        do_center_crop=False,  # Disable center cropping to match Keras behavior
    )

    # Use the preprocessor's crop size as the target resize size
    crop_size = hf_preprocessor.crop_size

    # Adjust the preprocessor's resize size to match crop size
    # This ensures that the resize operation will resize the image to the
    # target resolution (crop_size)
    hf_preprocessor.size = crop_size
    hf_model.eval()

    keras_backbone, hf_config = convert_model(hf_model)
    keras_image_converter = convert_image_converter(hf_preprocessor)
    keras_image_preprocessor = DeiTImageClassifierPreprocessor(
        image_converter=keras_image_converter
    )
    print("✅ KerasHub model loaded.")

    convert_backbone_weights(keras_backbone, hf_model)
    print("✅ Backbone weights converted.")

    if hf_config["architectures"][0] == "DeiTForImageClassificationWithTeacher":
        keras_model = DeiTImageClassifier(
            backbone=keras_backbone, num_classes=len(hf_config["id2label"])
        )
        convert_head_weights(keras_model, hf_model)
        print("✅ Head weights converted.")
        validate_output(
            keras_model,
            keras_image_converter,
            hf_model,
            hf_preprocessor,
            head_weights=True,
        )
        print("✅ Output validated.")
        keras_model.preprocessor = keras_image_preprocessor
        keras_model.save_to_preset(f"./{preset}")
    else:
        # access the backbone
        hf_model = hf_model.deit
        validate_output(
            keras_backbone,
            keras_image_converter,
            hf_model,
            hf_preprocessor,
        )
        print("✅ Output validated.")
        keras_backbone.save_to_preset(f"./{preset}")
        keras_image_preprocessor.save_to_preset(f"./{preset}")

    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
