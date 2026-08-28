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
}

MAX_NEW_TOKENS = 64

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


def run_hf_text_forward(hf_model, hf_preset):
    hf_inputs = build_text_inputs(hf_preset, TEXT_PROMPT)
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    return {
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().numpy(),
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
    hf_model.eval()

    text_results = run_hf_text_forward(hf_model, hf_preset)

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


def test_generate(
    label, keras_model, prompt, hf_generated_text, prompt_token_count, image
):
    x = {"prompts": [prompt], "images": [[image]]}
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
    print(f"✅ [{label}] Token IDs match.")


def validate_output(keras_model, hf_results, skip_generate=False):
    check_param_count(keras_model, hf_results)
    backbone = keras_model.backbone
    preprocessor = keras_model.preprocessor
    text_results = hf_results["text"]
    image_results = hf_results["image"]

    test_token_ids("text", preprocessor, TEXT_PROMPT, text_results["token_ids"])

    # The backbone always declares `pixel_values`/`image_sizes`/
    # `placeholder_indices` as graph inputs, so a text-only forward pass
    # feeds it empty-batched image tensors rather than omitting them —
    # this is a no-op through the image-merge layer.
    vision_encoder = backbone.vision_encoder
    patch_size = vision_encoder.patch_size
    token_ids = ops.convert_to_tensor(text_results["token_ids"].astype("int32"))
    backbone_inputs = {
        "token_ids": token_ids,
        "padding_mask": ops.ones_like(token_ids),
        "pixel_values": ops.zeros(
            (0, vision_encoder.num_channels, patch_size, patch_size),
            dtype="float32",
        ),
        "image_sizes": ops.zeros((0, 2), dtype="int32"),
        "placeholder_indices": ops.zeros((1, 0), dtype="int32"),
    }
    keras_logits = run_kh_forward(backbone, backbone_inputs)
    test_numerics("text", keras_logits, text_results["logits"])

    test_token_ids(
        "image",
        preprocessor,
        image_results["prompt"],
        image_results["token_ids"],
        image=image_results["image"],
    )

    # Feed HF's preprocessed `pixel_values` directly, rather than re-running
    # the Keras preprocessor, to avoid PIL vs `ops.image.resize` divergence.
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

    if not skip_generate:
        keras_model.compile(sampler="greedy")
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
    print("\n✅ Tests passed!")

    del keras_model
    gc.collect()
    keras_model = keras_hub.models.Mistral3CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print("\n✅ Saved the model preset in bfloat16")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
