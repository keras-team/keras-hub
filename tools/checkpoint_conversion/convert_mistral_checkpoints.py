"""Convert Mistral HuggingFace checkpoints to KerasHub preset format.

Handles both text-only Mistral checkpoints and multimodal Mistral3 checkpoints. 
The script auto-detects preset type from HF config and validates accordingly.

Usage:
    python tools/checkpoint_conversion/convert_mistral_checkpoints.py \
        --preset mistral_7b_en
"""

import gc
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import torch

device = torch.device("cpu")
torch.set_default_device(device)

from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from keras import ops  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import Mistral3ForConditionalGeneration  # noqa: E402
from transformers import MistralForCausalLM  # noqa: E402

import keras_hub  # noqa: E402

_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

PRESET_MAP = {
    # Text-only Mistral models.
    "mistral_7b_en": "mistralai/Mistral-7B-v0.1",
    "mistral_0.3_7b_en": "mistralai/Mistral-7B-v0.3",
    "mistral_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral_0.2_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_0.3_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.3",
    "magistral_small_2506_en": "mistralai/Magistral-Small-2506",
    "magistral_small_2507_en": "mistralai/Magistral-Small-2507",
    # Multimodal Mistral3 (Pixtral vision tower) models.
    "mistral_small_3.1_24b_base_2503_en": (
        "mistralai/Mistral-Small-3.1-24B-Base-2503"
    ),
    "mistral_small_3.1_24b_instruct_2503_en": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    ),
    "mistral_small_3.2_24b_instruct_2506_en": (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    ),
}

TEXT_PROMPT = "What is Keras?"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def is_multimodal_config(hf_config):
    return hasattr(hf_config, "vision_config")


def load_reference_image():
    return Image.open(requests.get(_IMAGE_URL, stream=True).raw).convert("RGB")


def build_multimodal_inputs(hf_config, hf_processor, image):
    image_token_index = hf_config.image_token_index

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]
    prompt = hf_processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = hf_processor(text=prompt, images=image, return_tensors="np")

    token_ids = inputs["input_ids"].astype("int32")
    padding_mask = inputs["attention_mask"].astype("int32")
    pixel_values = inputs["pixel_values"].astype("float32")
    image_sizes = inputs["image_sizes"].astype("int32")

    flat_ids = token_ids.reshape(-1)
    placeholder_indices = np.where(flat_ids == image_token_index)[0].astype(
        "int32"
    )[None, :]

    return {
        "prompt": prompt,
        "image": np.asarray(image),
        "token_ids": token_ids,
        "padding_mask": padding_mask,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "placeholder_indices": placeholder_indices,
    }


def precompute_hf_text_outputs(hf_preset):
    hf_model = MistralForCausalLM.from_pretrained(
        hf_preset, device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_inputs = hf_tokenizer([TEXT_PROMPT], return_tensors="pt")
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    hf_results = {
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().numpy(),
        "num_parameters": hf_model.num_parameters(),
    }
    del hf_model, hf_tokenizer
    gc.collect()
    return hf_results


def precompute_hf_multimodal_outputs(hf_preset, hf_config):
    hf_model = Mistral3ForConditionalGeneration.from_pretrained(
        hf_preset, device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_processor = AutoProcessor.from_pretrained(hf_preset)
    image = load_reference_image()
    inputs = build_multimodal_inputs(hf_config, hf_processor, image)
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=torch.tensor(inputs["token_ids"]),
            attention_mask=torch.tensor(inputs["padding_mask"]),
            pixel_values=torch.tensor(inputs["pixel_values"]),
            image_sizes=torch.tensor(inputs["image_sizes"]),
        )
    hf_results = {
        **inputs,
        "logits": hf_outputs.logits.detach().cpu().numpy(),
        "num_parameters": hf_model.num_parameters(),
    }
    del hf_model, hf_processor
    gc.collect()
    return hf_results


def check_param_count(keras_model, hf_results):
    keras_params = keras_model.backbone.count_params()
    hf_params = hf_results["num_parameters"]
    print(f"\nKerasHub params: {keras_params:,}")
    print(f"HF params:       {hf_params:,}")
    np.testing.assert_equal(keras_params, hf_params)


def test_numerics(keras_logits, hf_logits, atol):
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")
    abs_diff = np.abs(keras_logits - hf_logits)
    print("KerasHub logits:", keras_logits[0, 0, :5])
    print("HF logits:      ", hf_logits[0, 0, :5])
    print(f"Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"Logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=atol)
        print(f"-> Logits match! (atol={atol})")
    except AssertionError as err:
        matched_pct = 100 * np.mean(abs_diff <= atol)
        print(f"-> Logits mismatch (atol={atol}): {matched_pct:.2f}% matched")
        print(err.args[0])


def test_token_ids(keras_model, hf_results):
    # Runs `generate_preprocess` so multimodal inputs also exercise the
    # preprocessor's own image-placeholder expansion.
    hf_token_ids = hf_results["token_ids"]
    sequence_length = hf_token_ids.shape[1]
    if "placeholder_indices" not in hf_results:
        keras_inputs = keras_model.preprocessor.generate_preprocess(
            [TEXT_PROMPT], sequence_length=sequence_length
        )
    else:
        keras_inputs = keras_model.preprocessor.generate_preprocess(
            {
                "prompts": [hf_results["prompt"]],
                "images": [[hf_results["image"]]],
            },
            sequence_length=sequence_length,
        )
    keras_token_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    np.testing.assert_array_equal(keras_token_ids, hf_token_ids)
    print("-> Token IDs match.")


def validate_output(keras_model, hf_results):
    check_param_count(keras_model, hf_results)
    test_token_ids(keras_model, hf_results)

    multimodal = "placeholder_indices" in hf_results
    if multimodal:
        backbone = keras_model.backbone
        assert not backbone.text_only_model
        assert backbone.vision_encoder is not None
        keras_inputs = {
            "token_ids": ops.convert_to_tensor(
                hf_results["token_ids"].astype("int32")
            ),
            "padding_mask": ops.convert_to_tensor(
                hf_results["padding_mask"].astype("int32")
            ),
            "pixel_values": ops.convert_to_tensor(hf_results["pixel_values"]),
            "image_sizes": ops.convert_to_tensor(
                hf_results["image_sizes"].astype("int32")
            ),
            "placeholder_indices": ops.convert_to_tensor(
                hf_results["placeholder_indices"].astype("int32")
            ),
        }
    else:
        token_ids = ops.convert_to_tensor(
            hf_results["token_ids"].astype("int32")
        )
        keras_inputs = {
            "token_ids": token_ids,
            "padding_mask": ops.ones_like(token_ids),
        }

    keras_hidden = keras_model.backbone(keras_inputs)
    keras_logits = keras_model.backbone.token_embedding(
        keras_hidden, reverse=True
    )
    test_numerics(keras_logits, hf_results["logits"], atol=1e-3)


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    hf_config = AutoConfig.from_pretrained(hf_preset)
    multimodal = is_multimodal_config(hf_config)
    print(
        f"\nDetected {'multimodal (Mistral3)' if multimodal else 'text-only'} "
        f"checkpoint for `{hf_preset}`"
    )

    if multimodal:
        hf_results = precompute_hf_multimodal_outputs(hf_preset, hf_config)
    else:
        hf_results = precompute_hf_text_outputs(hf_preset)
    print("\n-> Huggingface model loaded and reference outputs computed")

    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("\n-> KerasHub model loaded")

    validate_output(keras_model, hf_results)
    print("\n-> Tests passed!")

    del keras_model
    gc.collect()
    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print("\n-> Saved the model preset in bfloat16")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
