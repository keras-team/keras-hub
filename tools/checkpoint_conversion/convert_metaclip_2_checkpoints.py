"""Convert MetaCLIP 2 checkpoints.

MetaCLIP 2 Worldwide models (non-distilled):
- ViT-H-14-quickgelu-worldwide: 224 resolution, ~986M params
- ViT-H-14-378-worldwide: 378 resolution, ~986M params
- ViT-bigG-14-worldwide: 224 resolution, ~1.4B params
- ViT-bigG-14-378-worldwide: 378 resolution, ~1.4B params

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_huge_patch14_224 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_224
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_huge_patch14_378 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_378
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_giant_patch14_224 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_224
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_giant_patch14_378 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_378
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

import keras_hub
from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.models.metaclip_2.metaclip_2_image_converter import (
    MetaCLIP2ImageConverter,
)
from keras_hub.src.models.metaclip_2.metaclip_2_preprocessor import (
    MetaCLIP2Preprocessor,
)
from keras_hub.src.models.metaclip_2.metaclip_2_text_encoder import (
    MetaCLIP2TextEncoder,
)
from keras_hub.src.models.metaclip_2.metaclip_2_tokenizer import (
    MetaCLIP2Tokenizer,
)
from keras_hub.src.models.metaclip_2.metaclip_2_vision_encoder import (
    MetaCLIP2VisionEncoder,
)

FLAGS = flags.FLAGS

# MetaCLIP 2 Worldwide models from Meta AI (non-distilled)
# These correspond to the models from the Meta CLIP 2 collection:
# https://huggingface.co/collections/facebook/meta-clip-2
PRESET_MAP = {
    # ViT-H-14-quickgelu-worldwide (224 resolution)
    "metaclip_2_vit_huge_patch14_224": "facebook/metaclip-2-worldwide-huge-quickgelu",
    # ViT-H-14-378-worldwide (378 resolution)
    "metaclip_2_vit_huge_patch14_378": "facebook/metaclip-2-worldwide-huge-378",
    # ViT-bigG-14-worldwide (224 resolution)
    "metaclip_2_vit_giant_patch14_224": "facebook/metaclip-2-worldwide-giant",
    # ViT-bigG-14-378-worldwide (378 resolution)
    "metaclip_2_vit_giant_patch14_378": "facebook/metaclip-2-worldwide-giant-378",
}

# XLM-V tokenizer used by MetaCLIP 2 for multilingual support
XLM_V_TOKENIZER = "facebook/xlm-v-base"

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
    """Convert HuggingFace MetaCLIP 2 model to Keras Hub model."""
    vision_encoder_config = hf_model.vision_model.config.to_dict()
    text_encoder_config = hf_model.text_model.config.to_dict()
    projection_dim = hf_model.config.to_dict().get("projection_dim", None)
    if projection_dim is None:
        assert (
            vision_encoder_config["projection_dim"]
            == text_encoder_config["projection_dim"]
        )
        projection_dim = vision_encoder_config["projection_dim"]
    image_size = vision_encoder_config["image_size"]

    # MetaCLIP 2 uses quick_gelu activation
    hidden_act = vision_encoder_config.get("hidden_act", "quick_gelu")

    vision_encoder = MetaCLIP2VisionEncoder(
        patch_size=vision_encoder_config["patch_size"],
        hidden_dim=vision_encoder_config["hidden_size"],
        num_layers=vision_encoder_config["num_hidden_layers"],
        num_heads=vision_encoder_config["num_attention_heads"],
        intermediate_dim=vision_encoder_config["intermediate_size"],
        intermediate_activation=hidden_act,
        image_shape=(image_size, image_size, 3),
    )
    text_encoder = MetaCLIP2TextEncoder(
        vocabulary_size=text_encoder_config["vocab_size"],
        embedding_dim=text_encoder_config["hidden_size"],
        hidden_dim=text_encoder_config["hidden_size"],
        num_layers=text_encoder_config["num_hidden_layers"],
        num_heads=text_encoder_config["num_attention_heads"],
        intermediate_dim=text_encoder_config["intermediate_size"],
        intermediate_activation=text_encoder_config.get(
            "hidden_act", "quick_gelu"
        ),
        max_sequence_length=text_encoder_config["max_position_embeddings"],
    )
    return MetaCLIP2Backbone(
        vision_encoder, text_encoder, projection_dim=projection_dim
    )


def convert_weights(keras_hub_model, hf_model):
    """Convert weights from HuggingFace to Keras Hub format."""
    # Get `state_dict` from `hf_model`.
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())  # Add buffers.

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
            f"{weight_key}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.q_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.k_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.v_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    # Port vision encoder.
    port_weights(
        keras_hub_model.vision_encoder.embedding.patch_embedding.kernel,
        "vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    port_weights(
        keras_hub_model.vision_encoder.embedding.position_embedding.embeddings,
        "vision_model.embeddings.position_embedding.weight",
    )
    port_weights(
        keras_hub_model.vision_encoder.embedding.class_embedding,
        "vision_model.embeddings.class_embedding",
    )
    port_weights(
        keras_hub_model.vision_encoder.embedding.position_ids,
        "vision_model.embeddings.position_ids",
    )
    port_ln(
        keras_hub_model.vision_encoder.pre_layer_norm,
        "vision_model.pre_layrnorm",
    )
    encoder_layers = keras_hub_model.vision_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "vision_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].attention,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm_1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm_2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].dense_1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].dense_2, f"{prefix}.{i}.mlp.fc2")
    port_ln(
        keras_hub_model.vision_encoder.post_layer_norm,
        "vision_model.post_layernorm",
    )
    port_dense(keras_hub_model.vision_projection, "visual_projection")

    # Port text encoder.
    port_weights(
        keras_hub_model.text_encoder.embedding.token_embedding._embeddings,
        "text_model.embeddings.token_embedding.weight",
    )
    port_weights(
        keras_hub_model.text_encoder.embedding.position_embedding.position_embeddings,
        "text_model.embeddings.position_embedding.weight",
    )
    encoder_layers = keras_hub_model.text_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "text_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].attention,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm_1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm_2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].dense_1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].dense_2, f"{prefix}.{i}.mlp.fc2")
    port_ln(
        keras_hub_model.text_encoder.layer_norm, "text_model.final_layer_norm"
    )
    port_dense(keras_hub_model.text_projection, "text_projection")

    # Port logit scale.
    port_weights(keras_hub_model.metaclip_2_head.logit_scale, "logit_scale")


def convert_image_converter(hf_image_processor):
    """Convert HuggingFace image processor to Keras Hub image converter."""
    config = hf_image_processor.to_dict()
    image_size = (config["crop_size"]["height"], config["crop_size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return MetaCLIP2ImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",  # MetaCLIP 2 defaults to bicubic resampling.
    )


def convert_tokenizer():
    """Convert XLM-V tokenizer to Keras Hub tokenizer.

    MetaCLIP 2 uses the XLM-V tokenizer (facebook/xlm-v-base) which is a
    SentencePiece-based multilingual tokenizer with ~901K vocabulary
    supporting 100+ languages.
    """
    # Download the SentencePiece model from XLM-V
    spm_path = hf_hub_download(
        XLM_V_TOKENIZER, "sentencepiece.bpe.model", token=True
    )

    # Read the SentencePiece proto
    with open(spm_path, "rb") as f:
        proto = f.read()

    return MetaCLIP2Tokenizer(proto=proto)


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    hf_model,
    hf_model_processor,
):
    """Validate that Keras Hub output matches HuggingFace output."""
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)
    text = ["a photo of a cat", "a photo of a dog"]

    # Preprocess with hf.
    hf_inputs = hf_model_processor(
        text=text,
        images=[image, image],
        return_tensors="pt",
        padding="max_length",
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
    hf_vision_logits = hf_outputs.logits_per_image.detach().cpu().numpy()

    # Call with keras.
    keras_preprocessor = MetaCLIP2Preprocessor(keras_tokenizer)
    token_ids = keras_preprocessor(
        {"images": keras_preprocessed, "prompts": text}
    )["token_ids"]
    keras_outputs = keras_model.predict(
        {"images": images, "token_ids": token_ids}, verbose=0
    )
    keras_vision_logits = keras.ops.convert_to_numpy(
        keras_outputs["vision_logits"]
    )

    print("🔶 Keras output:", keras_vision_logits[0, :10])
    print("🔶 HF output:", hf_vision_logits[0, :10])
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

    print(f"🏃 Converting {preset}")

    # Load huggingface model.
    hf_model = AutoModel.from_pretrained(hf_preset)
    hf_preprocessor = AutoProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    keras_model = convert_model(hf_model)
    keras_image_converter = convert_image_converter(
        hf_preprocessor.image_processor
    )
    keras_tokenizer = convert_tokenizer()
    print("✅ KerasHub model loaded.")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted.")

    validate_output(
        keras_model,
        keras_image_converter,
        keras_tokenizer,
        hf_model,
        hf_preprocessor,
    )
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_image_converter.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
