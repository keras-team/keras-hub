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
from huggingface_hub import hf_hub_download  # noqa: E402
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

IMAGE_PROMPT = "What is in this image?"
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def is_multimodal_config(hf_config):
    return hasattr(hf_config, "vision_config")


def load_reference_image():
    return Image.open(requests.get(_IMAGE_URL, stream=True).raw).convert("RGB")


def build_text_inputs(hf_preset, text):
    # Some checkpoints (e.g. Mistral Small 3.2) ship only `tekken.json`,
    # with no `tokenizer_config.json` for `AutoTokenizer` to resolve a
    # class from; fall back to `mistral_common` reading it directly.
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
        return hf_tokenizer([text], return_tensors="pt")
    except OSError:
        pass

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tekken_path = hf_hub_download(hf_preset, "tekken.json")
    raw_tokenizer = MistralTokenizer.from_file(
        tekken_path
    ).instruct_tokenizer.tokenizer
    token_ids = raw_tokenizer.encode(text, bos=True, eos=False)
    return {
        "input_ids": torch.tensor([token_ids]),
        "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
    }


def build_multimodal_inputs(hf_preset, hf_config, image):
    # Falls back to `mistral_common`'s `encode_chat_completion` when a
    # checkpoint has no chat template (e.g. a base model) or no
    # `preprocessor_config.json` for `AutoProcessor` to resolve a class
    # from (e.g. Mistral Small 3.2).
    image_token_index = hf_config.image_token_index

    try:
        hf_processor = AutoProcessor.from_pretrained(hf_preset)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": IMAGE_PROMPT},
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
    except (OSError, ValueError):
        from mistral_common.protocol.instruct.chunk import ImageChunk
        from mistral_common.protocol.instruct.chunk import TextChunk
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import (
            ChatCompletionRequest,
        )
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tekken_path = hf_hub_download(hf_preset, "tekken.json")
        request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageChunk(image=image),
                        TextChunk(text=IMAGE_PROMPT),
                    ]
                )
            ]
        )
        tokenized = MistralTokenizer.from_file(
            tekken_path
        ).encode_chat_completion(request)
        token_ids = np.array([tokenized.tokens], dtype="int32")
        padding_mask = np.ones_like(token_ids)
        pixel_values = tokenized.images[0][None, ...].astype("float32")
        image_sizes = np.array([pixel_values.shape[-2:]], dtype="int32")
        prompt = f"[INST][IMG]{IMAGE_PROMPT}[/INST]"

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


def run_hf_text_forward(hf_model, hf_preset):
    hf_inputs = build_text_inputs(hf_preset, TEXT_PROMPT)
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    return {
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().numpy(),
    }


def precompute_hf_outputs(hf_preset, hf_config, multimodal):
    model_cls = (
        Mistral3ForConditionalGeneration if multimodal else MistralForCausalLM
    )
    hf_model = model_cls.from_pretrained(
        hf_preset, device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_results = {
        "text": run_hf_text_forward(hf_model, hf_preset),
        "num_parameters": hf_model.num_parameters(),
    }

    if multimodal:
        image = load_reference_image()
        inputs = build_multimodal_inputs(hf_preset, hf_config, image)
        with torch.no_grad():
            hf_image_outputs = hf_model(
                input_ids=torch.tensor(inputs["token_ids"]),
                attention_mask=torch.tensor(inputs["padding_mask"]),
                pixel_values=torch.tensor(inputs["pixel_values"]),
                image_sizes=torch.tensor(inputs["image_sizes"]),
            )
        hf_results["image"] = {
            **inputs,
            "logits": hf_image_outputs.logits.detach().cpu().numpy(),
        }

    del hf_model
    gc.collect()
    return hf_results


def check_param_count(keras_model, hf_results):
    keras_params = keras_model.backbone.count_params()
    hf_params = hf_results["num_parameters"]
    print(f"\nKerasHub params: {keras_params:,}")
    print(f"HF params:       {hf_params:,}")
    np.testing.assert_equal(keras_params, hf_params)
    print("✅ Parameter count matches.")


def test_numerics(label, keras_logits, hf_logits):
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")
    abs_diff = np.abs(keras_logits - hf_logits)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"KerasHub logits [{label}]:", keras_logits[0, 0, :5])
    print(f"HF logits [{label}]:      ", hf_logits[0, 0, :5])
    try:
        np.testing.assert_allclose(
            keras_logits, hf_logits, atol=1e-3, rtol=1e-3
        )
        print(
            f"✅ [{label}] Logits within 1e-3 tolerance "
            f"(max={max_diff:.6f}, mean={mean_diff:.6f})."
        )
    except AssertionError:
        tol = 1e-3 + 1e-3 * np.abs(hf_logits)
        mismatched = int(np.sum(abs_diff > tol))
        total = hf_logits.size
        matched_pct = 100 * (1.0 - mismatched / total)
        print(
            f"⚠️  [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={matched_pct:.2f}% ({total - mismatched}/{total}).\n"
        )


def run_kh_forward(backbone, backbone_inputs):
    with torch.no_grad():
        hidden_states = backbone(backbone_inputs)
        return backbone.token_embedding(hidden_states, reverse=True)


def test_token_ids(label, preprocessor, prompt, hf_token_ids, image=None):
    x = {"prompts": [prompt]}
    if image is not None:
        x["images"] = [[image]]
    keras_inputs = preprocessor.generate_preprocess(
        x, sequence_length=hf_token_ids.shape[1]
    )
    keras_token_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    np.testing.assert_array_equal(keras_token_ids, hf_token_ids)
    print(f"✅ [{label}] Token IDs match.")


def validate_output(keras_model, hf_results):
    check_param_count(keras_model, hf_results)
    backbone = keras_model.backbone
    preprocessor = keras_model.preprocessor
    text_results = hf_results["text"]

    test_token_ids("text", preprocessor, TEXT_PROMPT, text_results["token_ids"])

    if "image" not in hf_results:
        return
    image_results = hf_results["image"]
    test_token_ids(
        "image",
        preprocessor,
        image_results["prompt"],
        image_results["token_ids"],
        image=image_results["image"],
    )

    token_ids = ops.convert_to_tensor(text_results["token_ids"].astype("int32"))
    backbone_inputs = {
        "token_ids": token_ids,
        "padding_mask": ops.ones_like(token_ids),
    }
    if not backbone.text_only_model:
        vision_encoder = backbone.vision_encoder
        patch_size = vision_encoder.patch_size
        backbone_inputs.update(
            {
                "pixel_values": ops.zeros(
                    (0, vision_encoder.num_channels, patch_size, patch_size),
                    dtype="float32",
                ),
                "image_sizes": ops.zeros((0, 2), dtype="int32"),
                "placeholder_indices": ops.zeros((1, 0), dtype="int32"),
            }
        )
    keras_logits = run_kh_forward(backbone, backbone_inputs)
    test_numerics("text", keras_logits, text_results["logits"])

    # Feed HF's preprocessed `pixel_values` directly, rather than
    # re-running the Keras preprocessor, to avoid PIL vs `ops.image.resize`
    # divergence.
    backbone_inputs = {
        "token_ids": ops.convert_to_tensor(
            image_results["token_ids"].astype("int32")
        ),
        "padding_mask": ops.convert_to_tensor(
            image_results["padding_mask"].astype("int32")
        ),
        "pixel_values": ops.convert_to_tensor(image_results["pixel_values"]),
        "image_sizes": ops.convert_to_tensor(
            image_results["image_sizes"].astype("int32")
        ),
        "placeholder_indices": ops.convert_to_tensor(
            image_results["placeholder_indices"].astype("int32")
        ),
    }
    keras_logits = run_kh_forward(backbone, backbone_inputs)
    test_numerics("image", keras_logits, image_results["logits"])


def main(_):
    if FLAGS.preset not in PRESET_MAP:
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

    hf_results = precompute_hf_outputs(hf_preset, hf_config, multimodal)
    print("\n-> Huggingface model loaded and reference outputs computed")

    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("\n-> KerasHub model loaded")

    validate_output(keras_model, hf_results)
    print("\n✅ Tests passed!")

    del keras_model
    gc.collect()
    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print("\n✅ Saved the model preset in bfloat16")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
