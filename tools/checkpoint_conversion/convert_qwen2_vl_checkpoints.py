"""Convert Qwen2-VL checkpoints from Hugging Face to KerasHub presets.

Uses the maintained `hf://` loading path (utils/transformers/convert_qwen2_vl)
for the actual conversion, then validates the result against the HF model:
parameter count, tokenization, text and multimodal logits, and generation.

    python tools/checkpoint_conversion/convert_qwen2_vl_checkpoints.py \
        --preset qwen2_vl_2b_instruct
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import gc  # noqa: E402
import shutil  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from keras import ops  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

PRESET_MAP = {
    "qwen2_vl_2b_base": "Qwen/Qwen2-VL-2B",
    "qwen2_vl_2b_instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_7b_base": "Qwen/Qwen2-VL-7B",
    "qwen2_vl_7b_instruct": "Qwen/Qwen2-VL-7B-Instruct",
}

TEXT_PROMPT = "What is the capital of France?"
MULTIMODAL_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|>"
    "<|vision_end|>Describe this image."
    "<|im_end|>\n<|im_start|>assistant\n"
)
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/"
    "documentation-images/resolve/main/transformers/tasks/car.jpg"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "Skip end-to-end generation checks (logits still compared).",
)


def _extract_response(text):
    """Strip the prompt/chat-template prefix from generated text."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()


def _load_test_image():
    path = keras.utils.get_file("qwen2_vl_test_car.jpg", IMAGE_URL)
    return Image.open(path).convert("RGB").crop((0, 0, 616, 476))


def _count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(ops.convert_to_numpy(w).size for w in unique)


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset):
    """Run all HF forward passes and generation, returning numpy results."""
    results = {}

    # --- Text-only outputs ---
    hf_ids = hf_tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    results["text_token_ids"] = hf_ids
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device)
        )
    results["text_logits"] = hf_out.logits.detach().cpu().float().numpy()

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_gen = hf_model.generate(
                input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device),
                max_new_tokens=32,
                do_sample=False,
                repetition_penalty=1.0,
            )
        results["text_generated"] = hf_tokenizer.decode(
            hf_gen[0], skip_special_tokens=True
        )

    # --- Multimodal outputs ---
    raw_image = _load_test_image()
    # AutoImageProcessor, not AutoProcessor: transformers builds a video
    # processor for the latter, which requires torchvision.
    image_processor = AutoImageProcessor.from_pretrained(hf_preset)
    vision = image_processor(images=[raw_image], return_tensors="pt")
    grid_thw = vision["image_grid_thw"][0]
    # Expand the marker to the image's token count, as HF's processor does.
    pad = "<|image_pad|>"
    merge_len = image_processor.merge_size**2
    n_image_tokens = int(grid_thw.prod().item()) // merge_len
    prompt = MULTIMODAL_PROMPT.replace(pad, pad * n_image_tokens)
    enc = hf_tokenizer(prompt, return_tensors="pt")
    # transformers>=5.x needs mm_token_type_ids for M-RoPE (0=text, 1=image).
    image_pad_id = hf_tokenizer.convert_tokens_to_ids(pad)
    mm_token_type_ids = np.where(enc["input_ids"].numpy() == image_pad_id, 1, 0)
    hf_inputs = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "pixel_values": vision["pixel_values"].to(device),
        "image_grid_thw": vision["image_grid_thw"].to(device),
        "mm_token_type_ids": torch.tensor(
            mm_token_type_ids, dtype=torch.long
        ).to(device),
    }
    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)
    results["mm_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["mm_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    results["mm_attention_mask"] = (
        hf_inputs["attention_mask"].cpu().numpy().astype(np.int32)
    )
    results["mm_pixel_values"] = hf_inputs["pixel_values"].cpu().float().numpy()
    results["mm_image_grid_thw"] = (
        hf_inputs["image_grid_thw"].cpu().numpy().astype(np.int32)
    )
    results["raw_image"] = raw_image

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_gen = hf_model.generate(
                **hf_inputs,
                max_new_tokens=32,
                do_sample=False,
                repetition_penalty=1.0,
            )
        results["mm_generated"] = hf_tokenizer.decode(
            hf_gen[0], skip_special_tokens=True
        )

    return results


# ---------------------------------------------------------------
# 2. Parameter count
# ---------------------------------------------------------------
def test_parameter_count(keras_backbone, hf_param_count):
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)
    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")
    assert keras_params == hf_param_count, (
        f"Parameter count mismatch: keras={keras_params}, hf={hf_param_count}"
    )
    print("  Parameter counts match.")


# ---------------------------------------------------------------
# 3. Text-only validation
# ---------------------------------------------------------------
def validate_text_output(keras_model, hf_results):
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION")
    print("=" * 50)

    hf_ids = hf_results["text_token_ids"]
    keras_preprocessed = keras_model.preprocessor.generate_preprocess(
        TEXT_PROMPT, sequence_length=hf_ids.shape[1]
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
    keras_valid = keras_ids[keras_mask.astype(bool)]
    np.testing.assert_array_equal(keras_valid, hf_ids[0])
    print("  Token IDs match.")

    keras_logits = ops.convert_to_numpy(keras_model(keras_preprocessed)).astype(
        np.float32
    )

    hf_logits = hf_results["text_logits"]
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
    print("  Logits match within atol=1e-3.")

    if not FLAGS.skip_generation:
        keras_output = keras_model.generate(
            TEXT_PROMPT, max_length=int(hf_ids.shape[1]) + 32
        )
        keras_text = _extract_response(keras_output)
        hf_text = _extract_response(hf_results["text_generated"])
        print(f"  KerasHub: {keras_text}")
        print(f"  HF:       {hf_text}")
        assert keras_text == hf_text, (
            f"Greedy generation mismatch:\n  keras: {keras_text}\n"
            f"  hf:    {hf_text}"
        )
        print("  Greedy generation matches.")


def validate_multimodal_output(keras_model, hf_results):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping multimodal validation (no vision encoder).")
        return

    print("\n" + "=" * 50)
    print("MULTIMODAL VALIDATION")
    print("=" * 50)

    token_ids_np = hf_results["mm_input_ids"].astype(np.int32)

    # HF pixel_values is (N, C*T*pH*pW) flattened; KerasHub uses
    # (N, T, pH, pW, C).
    ve = keras_model.backbone.vision_encoder
    C, T = ve.in_channels, ve.temporal_patch_size
    pH = pW = ve.patch_size
    pixel_values_np = hf_results["mm_pixel_values"].reshape(-1, C, T, pH, pW)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 4, 1))

    preprocessed = keras_model.preprocessor.generate_preprocess(
        {
            "prompts": [MULTIMODAL_PROMPT],
            "images": [np.asarray(hf_results["raw_image"])],
        }
    )
    keras_pixels = ops.convert_to_numpy(preprocessed["pixel_values"])
    np.testing.assert_allclose(keras_pixels, pixel_values_np, atol=1e-5)
    print("  Preprocessor pixels match HF image processor (atol=1e-5).")

    keras_logits = ops.convert_to_numpy(keras_model(preprocessed)).astype(
        np.float32
    )
    keras_last = keras_logits[:, -1]
    hf_last = hf_results["mm_logits"][:, -1]
    abs_diff = np.abs(keras_last - hf_last)
    print(f"  Last-position logit mean diff: {abs_diff.mean():.6f}")
    print(f"  Last-position logit max diff:  {abs_diff.max():.6f}")
    np.testing.assert_allclose(keras_last, hf_last, atol=1e-2)
    print("  Logits match within atol=1e-2.")

    if not FLAGS.skip_generation:
        keras_output = keras_model.generate(
            {
                "prompts": [MULTIMODAL_PROMPT],
                "images": [np.array(hf_results["raw_image"])],
            },
            max_length=int(token_ids_np.shape[1]) + 32,
        )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        keras_text = _extract_response(keras_text)
        hf_text = _extract_response(hf_results["mm_generated"])
        print(f"  KerasHub output: {keras_text}")
        print(f"  HF output:       {hf_text}")
        assert keras_text == hf_text, (
            f"Greedy multimodal generation mismatch:\n  keras: {keras_text}\n"
            f"  hf:    {hf_text}"
        )
        print("  Greedy generation matches.")


# ---------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------
def save_preset(keras_model, preset_name):
    preset_path = f"./{preset_name}"
    if os.path.exists(preset_path):
        print(f"\n-> Removing existing preset at {preset_path}...")
        shutil.rmtree(preset_path)
    print(f"-> Saving KerasHub preset to {preset_path}...")
    keras_model.save_to_preset(preset_path)
    print(f"  Preset saved to {preset_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    # --- Phase 1: Load HF model and precompute all outputs ---
    print("-> Loading HF model...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset, torch_dtype=torch.float32
    ).to(device)
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset)

    # --- Phase 2: Free HF model, load KerasHub model ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.Qwen2VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    keras_model.compile(sampler="greedy")

    # --- Phase 3: Validate ---
    test_parameter_count(keras_model.backbone, hf_params)
    validate_text_output(keras_model, hf_results)
    validate_multimodal_output(keras_model, hf_results)

    # --- Phase 4: Save ---
    del keras_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n-> Reloading model in bfloat16 for saving...")
    save_model = keras_hub.models.Qwen2VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    save_preset(save_model, preset)
    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
