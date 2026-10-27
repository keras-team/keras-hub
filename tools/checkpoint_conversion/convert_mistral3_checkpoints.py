"""Convert multimodal Mistral3 HuggingFace checkpoints to KerasHub presets.

Usage:
    python tools/checkpoint_conversion/convert_mistral3_checkpoints.py \
        --preset mistral_small_3.1_24b_instruct_2503_en
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
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import Mistral3ForConditionalGeneration  # noqa: E402
from transformers import PixtralImageProcessor  # noqa: E402

import keras_hub  # noqa: E402

_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

PRESET_MAP = {
    "mistral_small_3.1_24b_base_2503_en": (
        "mistralai/Mistral-Small-3.1-24B-Base-2503"
    ),
    "mistral_small_3.1_24b_instruct_2503_en": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    ),
    "mistral_small_3.2_24b_instruct_2506_en": (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    ),
    "ministral_3_3b_base_2512_en": "mistralai/Ministral-3-3B-Base-2512",
    "ministral_3_3b_instruct_2512_en": (
        "mistralai/Ministral-3-3B-Instruct-2512"
    ),
    "ministral_3_3b_reasoning_2512_en": (
        "mistralai/Ministral-3-3B-Reasoning-2512"
    ),
    "ministral_3_8b_base_2512_en": "mistralai/Ministral-3-8B-Base-2512",
    "ministral_3_8b_instruct_2512_en": (
        "mistralai/Ministral-3-8B-Instruct-2512"
    ),
    "ministral_3_8b_reasoning_2512_en": (
        "mistralai/Ministral-3-8B-Reasoning-2512"
    ),
    "ministral_3_14b_base_2512_en": "mistralai/Ministral-3-14B-Base-2512",
    "ministral_3_14b_instruct_2512_en": (
        "mistralai/Ministral-3-14B-Instruct-2512"
    ),
    "ministral_3_14b_reasoning_2512_en": (
        "mistralai/Ministral-3-14B-Reasoning-2512"
    ),
    "shieldstral_1.0_3b_en": "mistralai/Shieldstral-1.0-3B",
    "magistral_small_2509_en": "mistralai/Magistral-Small-2509",
    "devstral_small_2_24b_instruct_2512_en": (
        "mistralai/Devstral-Small-2-24B-Instruct-2512"
    ),
}

MAX_NEW_TOKENS = 64

# Fraction of logits that may exceed the 1e-3 tolerance. Float32 noise in a
# deep model leaves a few outliers. A systematic error moves far more logits.
MAX_OUTLIER_FRACTION = 1e-6

TEXT_PROMPT = "What is Keras?"

IMAGE_PROMPT = "What is in this image?"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_boolean(
    "skip_generate",
    False,
    "Skip the generation comparison step. Useful for large models where "
    "generation is slow or unnecessary (numerics verification is sufficient).",
)


def load_reference_image():
    return Image.open(requests.get(_IMAGE_URL, stream=True).raw).convert("RGB")


def build_text_inputs(hf_preset, text):
    # Some checkpoints (e.g. Mistral Small 3.2) ship only `tekken.json`,
    # with no `tokenizer_config.json` for `AutoTokenizer` to resolve a
    # class from; fall back to `mistral_common` reading it directly.
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
        return hf_tokenizer([text], return_tensors="pt"), hf_tokenizer
    except OSError:
        pass

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tekken_path = hf_hub_download(hf_preset, "tekken.json")
    raw_tokenizer = MistralTokenizer.from_file(
        tekken_path
    ).instruct_tokenizer.tokenizer
    token_ids = raw_tokenizer.encode(text, bos=True, eos=False)
    inputs = {
        "input_ids": torch.tensor([token_ids]),
        "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
    }
    return inputs, raw_tokenizer


def run_hf_text_forward(hf_model, hf_preset, skip_generate=False):
    hf_inputs, hf_tokenizer = build_text_inputs(hf_preset, TEXT_PROMPT)
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    results = {
        "prompt": TEXT_PROMPT,
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().numpy(),
    }
    if not skip_generate:
        with torch.no_grad():
            generated_ids = hf_model.generate(
                **hf_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        prompt_length = hf_inputs["input_ids"].shape[1]
        generated_token_ids = generated_ids[0, prompt_length:].tolist()
        try:
            results["generated_text"] = hf_tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )
        except TypeError:
            # `mistral_common`'s raw tokenizer (used in the fallback path
            # above) doesn't accept `skip_special_tokens`.
            results["generated_text"] = hf_tokenizer.decode(generated_token_ids)
    return results


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
        # Mistral's chat template renders `bos_token` as literal text, so
        # `prompt` starts with `"<s>"`. `hf_processor` recognizes that text
        # and folds it into a single BOS id, but KerasHub's preprocessor
        # always prepends its own BOS via `add_start_token`; passing the
        # literal `"<s>"` through would double it. Strip it so `prompt`
        # matches the `except` branch's convention below (no literal BOS).
        bos_token = hf_processor.tokenizer.bos_token
        if bos_token and prompt.startswith(bos_token):
            prompt = prompt[len(bos_token) :]
        token_ids = inputs["input_ids"].astype("int32")
        padding_mask = inputs["attention_mask"].astype("int32")
        # Some chat templates (e.g. Ministral 3 Reasoning) render a literal
        # `"<s>"`, and `hf_processor` also adds its own BOS. Drop the
        # duplicate so the ids match KerasHub's single BOS.
        bos_id = hf_processor.tokenizer.bos_token_id
        if (
            bos_id is not None
            and token_ids.shape[1] > 1
            and np.all(token_ids[:, :2] == bos_id)
        ):
            token_ids = token_ids[:, 1:]
            padding_mask = padding_mask[:, 1:]
        pixel_values = inputs["pixel_values"].astype("float32")
        image_sizes = inputs["image_sizes"].astype("int32")
        tokenizer = hf_processor.tokenizer
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
        mistral_tokenizer = MistralTokenizer.from_file(tekken_path)
        tokenized = mistral_tokenizer.encode_chat_completion(request)
        token_ids = np.array([tokenized.tokens], dtype="int32")
        padding_mask = np.ones_like(token_ids)
        pixel_values = tokenized.images[0][None, ...].astype("float32")
        image_sizes = np.array([pixel_values.shape[-2:]], dtype="int32")
        # `mistral_common` resizes differently from HF. Take the pixels from
        # HF's `PixtralImageProcessor` instead. A checkpoint with no processor
        # config gets one built from the converter's defaults. Keep the
        # `mistral_common` pixels if the image size differs, because the
        # token ids depend on that size.
        patch = hf_config.vision_config.patch_size
        patch *= hf_config.spatial_merge_size
        try:
            image_processor = AutoProcessor.from_pretrained(
                hf_preset
            ).image_processor
        except (OSError, ValueError):
            from mistral_common.tokens.tokenizers.image import DATASET_MEAN
            from mistral_common.tokens.tokenizers.image import DATASET_STD

            image_processor = PixtralImageProcessor(
                image_mean=list(DATASET_MEAN),
                image_std=list(DATASET_STD),
                rescale_factor=1 / 255,
                size={"longest_edge": hf_config.vision_config.image_size},
            )
        hf_image_inputs = image_processor(
            images=image,
            patch_size={"height": patch, "width": patch},
            return_tensors="np",
        )
        hf_image_sizes = np.array(hf_image_inputs["image_sizes"], dtype="int32")
        if np.array_equal(hf_image_sizes, image_sizes):
            pixel_values = hf_image_inputs["pixel_values"].astype("float32")
        prompt = f"[INST][IMG]{IMAGE_PROMPT}[/INST]"
        tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

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
        "tokenizer": tokenizer,
    }


def precompute_hf_outputs(hf_preset, hf_config, skip_generate=False):
    hf_model = Mistral3ForConditionalGeneration.from_pretrained(
        hf_preset, device_map="cpu", torch_dtype=torch.float32
    )
    # FP8 checkpoints (e.g. Ministral 3, Devstral Small 2) dequantize to
    # bfloat16 regardless of `torch_dtype`. Cast them to match the rest.
    hf_model = hf_model.float()
    hf_model.eval()

    text_results = run_hf_text_forward(
        hf_model, hf_preset, skip_generate=skip_generate
    )

    image = load_reference_image()
    inputs = build_multimodal_inputs(hf_preset, hf_config, image)
    with torch.no_grad():
        hf_image_outputs = hf_model(
            input_ids=torch.tensor(inputs["token_ids"]),
            attention_mask=torch.tensor(inputs["padding_mask"]),
            pixel_values=torch.tensor(inputs["pixel_values"]),
            image_sizes=torch.tensor(inputs["image_sizes"]),
        )
    hf_results = {
        "num_parameters": hf_model.num_parameters(),
        "text": text_results,
        "image": {
            **inputs,
            "logits": hf_image_outputs.logits.detach().cpu().numpy(),
        },
    }
    if not skip_generate:
        with torch.no_grad():
            generated_ids = hf_model.generate(
                input_ids=torch.tensor(inputs["token_ids"]),
                attention_mask=torch.tensor(inputs["padding_mask"]),
                pixel_values=torch.tensor(inputs["pixel_values"]),
                image_sizes=torch.tensor(inputs["image_sizes"]),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        prompt_length = inputs["token_ids"].shape[1]
        generated_token_ids = generated_ids[0, prompt_length:].tolist()
        try:
            generated_text = inputs["tokenizer"].decode(
                generated_token_ids, skip_special_tokens=True
            )
        except TypeError:
            # `mistral_common`'s raw tokenizer (used in the fallback path
            # above) doesn't accept `skip_special_tokens`.
            generated_text = inputs["tokenizer"].decode(generated_token_ids)
        hf_results["image"]["generated_text"] = generated_text

    del hf_model
    gc.collect()
    return hf_results


def check_param_count(keras_model, hf_results):
    keras_params = keras_model.backbone.count_params()
    hf_params = hf_results["num_parameters"]
    print(f"\nKerasHub params: {keras_params:,}")
    print(f"HF params:       {hf_params:,}")
    np.testing.assert_equal(keras_params, hf_params)
    print("\n✓ Parameter count matches.")


def test_numerics(label, keras_logits, hf_logits):
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")
    abs_diff = np.abs(keras_logits - hf_logits)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"KerasHub logits [{label}]:", keras_logits[0, 0, :5])
    print(f"HF logits [{label}]:      ", hf_logits[0, 0, :5])
    tol = 1e-3 + 1e-3 * np.abs(hf_logits)
    mismatched = int(np.sum(abs_diff > tol))
    total = hf_logits.size
    summary = (
        f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
        f"outliers={mismatched}/{total}"
    )
    if mismatched / total <= MAX_OUTLIER_FRACTION:
        print(f"✓ [{label}] Logits within 1e-3 tolerance ({summary}).")
    else:
        print(f"✗ [{label}] Logits exceed 1e-3 tolerance — {summary}.\n")
        raise AssertionError(
            f"[{label}] {mismatched / total:.2e} of the logits exceed the "
            f"1e-3 tolerance. The limit is {MAX_OUTLIER_FRACTION:.0e}."
        )


def test_generate(
    label,
    keras_model,
    prompt,
    hf_generated_text,
    prompt_token_count,
    image=None,
):
    x = {"prompts": [prompt]}
    if image is not None:
        x["images"] = [[image]]
    max_length = prompt_token_count + MAX_NEW_TOKENS
    kh_output = keras_model.generate(x, max_length=max_length)
    kh_text = kh_output[0] if isinstance(kh_output, list) else kh_output
    if isinstance(kh_text, str):
        if kh_text.startswith(prompt):
            kh_text = kh_text[len(prompt) :]
        else:
            # `[IMG]` placeholders expand into real image tokens during
            # preprocessing and decode back to nothing, so the decoded text
            # won't literally start with `prompt` for image inputs. Strip
            # everything up to and including the last `[/INST]` instead.
            idx = kh_text.rfind("[/INST]")
            if idx != -1:
                kh_text = kh_text[idx + len("[/INST]") :]
    print(f"\n[{label}] HF generated: {hf_generated_text}")
    print(f"[{label}] KH generated: {kh_text}")


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
    print(f"✓ [{label}] Token IDs match.")


def validate_output(keras_model, hf_results, skip_generate=False):
    print("\n--- Parameter Count ---")
    check_param_count(keras_model, hf_results)
    backbone = keras_model.backbone
    preprocessor = keras_model.preprocessor
    text_results = hf_results["text"]
    image_results = hf_results["image"]

    print("\n--- Token IDs ---")
    test_token_ids("text", preprocessor, TEXT_PROMPT, text_results["token_ids"])
    test_token_ids(
        "image",
        preprocessor,
        image_results["prompt"],
        image_results["token_ids"],
        image=image_results["image"],
    )

    # The preprocessor adds the zero-length image inputs that the backbone
    # declares as graph inputs. The HF token ids have no end token.
    add_end_token = preprocessor.add_end_token
    preprocessor.add_end_token = False
    text_inputs, _, _ = preprocessor(
        {"prompts": [TEXT_PROMPT]},
        sequence_length=text_results["token_ids"].shape[1],
    )
    image_inputs, _, _ = preprocessor(
        {
            "prompts": [image_results["prompt"]],
            "images": [[image_results["image"]]],
        },
        sequence_length=image_results["token_ids"].shape[1],
    )
    preprocessor.add_end_token = add_end_token

    print("\n--- Numerics Comparison ---")
    keras_logits = run_kh_forward(backbone, text_inputs)
    test_numerics("text", keras_logits, text_results["logits"])
    keras_logits = run_kh_forward(backbone, image_inputs)
    test_numerics("image", keras_logits, image_results["logits"])

    if not skip_generate:
        keras_model.compile(sampler="greedy")
        print("\n--- Text Generation ---")
        test_generate(
            "text",
            keras_model,
            text_results["prompt"],
            text_results.get("generated_text"),
            text_results["token_ids"].shape[1],
        )
        test_generate(
            "image",
            keras_model,
            image_results["prompt"],
            image_results.get("generated_text"),
            image_results["token_ids"].shape[1],
            image=image_results["image"],
        )


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(hf_preset)

    hf_results = precompute_hf_outputs(
        hf_preset, hf_config, skip_generate=FLAGS.skip_generate
    )
    print("\n-> Huggingface model loaded and reference outputs computed")

    keras_model = keras_hub.models.Mistral3CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("\n-> KerasHub model loaded")

    validate_output(keras_model, hf_results, skip_generate=FLAGS.skip_generate)
    print("\n✓ Tests passed!")

    del keras_model
    gc.collect()
    keras_model = keras_hub.models.Mistral3CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print("\n✓ Saved the model preset in bfloat16")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
