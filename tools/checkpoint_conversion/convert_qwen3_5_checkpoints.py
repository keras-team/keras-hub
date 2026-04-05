"""Convert Qwen3.5 HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_5_checkpoints.py \
        --preset qwen3_5_0.8b_base
"""

import gc
import os
import random
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "qwen3_5_0.8b_base": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3_5_0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_2b_base": "Qwen/Qwen3.5-2B-Base",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b_base": "Qwen/Qwen3.5-4B-Base",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_9b_base": "Qwen/Qwen3.5-9B-Base",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    "qwen3_5_27b": "Qwen/Qwen3.5-27B",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEXT_PROMPT = "What is Keras?"
MULTIMODAL_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset):
    """Precompute all HF outputs needed for validation.

    Runs all HF forward passes and generation, returning results as
    numpy arrays. The HF model can then be deleted to free memory.
    """
    results = {}

    # --- Text-only outputs ---
    hf_ids = hf_tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    results["text_token_ids"] = hf_ids

    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device),
        )
    results["text_logits"] = hf_out.logits.detach().cpu().float().numpy()

    with torch.no_grad():
        hf_gen = hf_model.generate(
            input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device),
            max_new_tokens=32,
            do_sample=False,
        )
    results["text_generated"] = hf_tokenizer.decode(
        hf_gen[0], skip_special_tokens=True
    )

    # --- Multimodal outputs ---
    raw_image = _load_test_image()
    processor = AutoProcessor.from_pretrained(hf_preset)
    hf_inputs = processor(
        text=[MULTIMODAL_PROMPT], images=[raw_image], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)
    results["mm_logits"] = hf_out.logits.detach().cpu().float().numpy()

    print(f"   HF pixel_values shape: {hf_inputs['pixel_values'].shape}")

    results["mm_input_ids"] = (
        hf_inputs["input_ids"].cpu().numpy().astype(np.int32)
    )
    results["mm_attention_mask"] = (
        hf_inputs["attention_mask"].cpu().numpy().astype(np.int32)
    )
    results["mm_pixel_values"] = hf_inputs["pixel_values"].cpu().float().numpy()
    results["mm_image_grid_thw"] = (
        hf_inputs["image_grid_thw"].cpu().numpy().astype(np.int32)
    )

    with torch.no_grad():
        hf_gen = hf_model.generate(
            **hf_inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    results["mm_generated"] = processor.batch_decode(
        hf_gen, skip_special_tokens=True
    )[0]
    results["raw_image"] = raw_image

    return results


# ---------------------------------------------------------------
# 2. Parameter count comparison
# ---------------------------------------------------------------
def test_parameter_count(keras_backbone, hf_param_count):
    """Compare parameter counts between KerasHub and HF models."""
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)

    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")

    if keras_params == hf_param_count:
        print("  ✓ Parameter counts match!")
    else:
        diff = hf_param_count - keras_params
        print(
            f"  Parameter count difference: {diff:,} "
            f"(expected — MTP training heads excluded from KerasHub)"
        )


# ---------------------------------------------------------------
# 3. Validate text-only output
# ---------------------------------------------------------------
def validate_text_output(keras_model, hf_results):
    """Validate text-only tokenization, logits, and generation."""
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION")
    print("=" * 50)

    hf_ids = hf_results["text_token_ids"]

    # --- Token ID parity ---
    keras_preprocessed = keras_model.preprocessor.generate_preprocess(
        TEXT_PROMPT, sequence_length=hf_ids.shape[1]
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
    keras_valid = keras_ids[keras_mask.astype(bool)]

    print(f"\n  HF token ids:      {hf_ids[0][:10].tolist()}")
    print(f"  KerasHub token ids: {keras_valid[:10].tolist()}")
    np.testing.assert_array_equal(keras_valid, hf_ids[0])
    print("  ✓ Token IDs match.")

    # --- Logit comparison (preprocessor-free forward pass) ---
    token_ids = ops.convert_to_tensor(hf_ids.astype(np.int32))
    padding_mask = ops.ones_like(token_ids)

    keras_hidden = keras_model.backbone(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    keras_logits = keras_model.backbone.token_embedding(
        keras_hidden, reverse=True
    )
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)

    hf_logits = hf_results["text_logits"]
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-4)
        print("  ✓ Logits match within atol=1e-4.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-4: {e}")

    # --- End-to-end generation ---
    print("\n  Generating text...")
    keras_output = keras_model.generate(TEXT_PROMPT, max_length=32)
    print(f"  KerasHub: {keras_output}")
    print(f"  HF:       {hf_results['text_generated']}")
    print("  ✓ Text generation completed.")


# ---------------------------------------------------------------
# 4. Validate multimodal output
def validate_multimodal_output(keras_model, hf_results):
    """Validate multimodal logits and generation."""
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping multimodal validation (no vision encoder).")
        return

    print("\n" + "=" * 50)
    print("MULTIMODAL VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    token_ids_np = hf_results["mm_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["mm_attention_mask"])

    # HF pixel_values: (N, C*T*pH*pW) flattened → KerasHub 5D: (N,T,pH,pW,C).
    pixel_values_np = hf_results["mm_pixel_values"]
    ve = backbone.vision_encoder
    C, T = ve.in_channels, ve.temporal_patch_size
    pH = pW = ve.patch_size
    pixel_values_np = pixel_values_np.reshape(-1, C, T, pH, pW)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 4, 1))
    pixel_values = ops.convert_to_tensor(pixel_values_np)
    image_grid_thw = ops.convert_to_tensor(hf_results["mm_image_grid_thw"])

    # Vision indices from HF's token_ids.
    image_pad_id = keras_model.preprocessor.image_token_id
    vision_pos = np.where(token_ids_np[0] == image_pad_id)[0]
    vision_indices = ops.convert_to_tensor(vision_pos.astype(np.int32))

    # --- Forward pass ---
    img_embeds = backbone.vision_encoder(pixel_values, image_grid_thw)
    x = backbone.token_embedding(token_ids)
    x = backbone.interleave_embeddings(
        image_embeddings=img_embeds,
        text_embeddings=x,
        vision_indices=vision_indices,
    )

    position_ids = keras_model.preprocessor._compute_position_ids(
        token_ids, image_grid_thw
    )
    position_ids = ops.convert_to_tensor(position_ids)

    for layer in backbone.transformer_layers:
        x = layer(
            x,
            decoder_padding_mask=padding_mask,
            position_ids=position_ids,
        )

    x = backbone.layer_norm(x)
    keras_logits = backbone.token_embedding(x, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["mm_logits"]

    # --- Logit comparison ---
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-4)
        print("  ✓ Logits match within atol=1e-4.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-4: {e}")

    # --- End-to-end generation ---
    print(f"\n  HF output: {hf_results['mm_generated']}")

    raw_image = hf_results["raw_image"]
    keras_output = keras_model.generate(
        {"prompts": [MULTIMODAL_PROMPT], "images": [np.array(raw_image)]},
        max_length=2048,
    )
    keras_text = (
        keras_output[0] if isinstance(keras_output, list) else keras_output
    )
    print(f"  KerasHub output: {keras_text}")
    print("  ✓ Multimodal generation completed.")


# ---------------------------------------------------------------
# 5. Save preset
# ---------------------------------------------------------------
def save_preset(keras_model, preset_name):
    """Save the converted model as a KerasHub preset."""
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


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
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset)
    hf_results["hf_param_count"] = hf_params
    print("   HF outputs precomputed!")

    # --- Phase 2: Free HF model to reclaim memory ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   HF model released.")

    # --- Phase 3: Load KerasHub model ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.Qwen3_5CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate against precomputed HF outputs ---
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_text_output(keras_model, hf_results)
    validate_multimodal_output(keras_model, hf_results)

    # --- Phase 5: Save preset ---
    save_preset(keras_model, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
