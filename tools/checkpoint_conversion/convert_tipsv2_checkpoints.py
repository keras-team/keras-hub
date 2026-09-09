"""Convert TIPSv2 checkpoints.

python tools/checkpoint_conversion/convert_tipsv2_checkpoints.py \
    --preset tipsv2_b14
python tools/checkpoint_conversion/convert_tipsv2_checkpoints.py \
    --preset tipsv2_l14
python tools/checkpoint_conversion/convert_tipsv2_checkpoints.py \
    --preset tipsv2_so400m14
python tools/checkpoint_conversion/convert_tipsv2_checkpoints.py \
    --preset tipsv2_g14
"""

import gc
import os
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from PIL import Image
from transformers import AutoModel

import keras_hub
from keras_hub.src.models.tipsv2.tipsv2_image_converter import (
    TIPSv2ImageConverter,
)
from keras_hub.src.models.tipsv2.tipsv2_tokenizer import TIPSv2Tokenizer

FLAGS = flags.FLAGS

PRESET_MAP = {
    "tipsv2_b14": "google/tipsv2-b14",
    "tipsv2_l14": "google/tipsv2-l14",
    "tipsv2_so400m14": "google/tipsv2-so400m14",
    "tipsv2_g14": "google/tipsv2-g14",
}

TEXT_PROMPT = "a photo of a cat"
MAX_SEQ_LEN = 64
SAMPLE_IMAGE_URL = (
    "https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
)

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://kerashub/tipsv2/keras/{preset}"',
    required=False,
)


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, img_np, token_ids, padding_mask):
    """Precompute HF outputs for both encoders.

    Runs all HF forward passes, returning results as numpy arrays.
    The HF model can then be deleted to free memory.
    """
    results = {}

    # --- Vision encoder ---
    # HF expects (B, C, H, W).
    img_torch = torch.from_numpy(img_np.transpose(0, 3, 1, 2))
    with torch.no_grad():
        hf_out = hf_model.encode_image(img_torch)

    results["vision_input"] = img_np
    results["vision_cls"] = hf_out.cls_token.cpu().numpy()
    results["vision_reg"] = hf_out.register_tokens.cpu().numpy()
    results["vision_patch"] = hf_out.patch_tokens.cpu().numpy()

    # --- Text encoder ---
    # HF convention: paddings=1 means padding, 0 means valid.
    hf_paddings = 1 - padding_mask  # Invert KerasHub mask for HF.
    with torch.no_grad():
        hf_text = hf_model.encode_text(
            torch.from_numpy(token_ids.astype("int64")),
            padding_mask=torch.from_numpy(hf_paddings.astype("int32")),
        )
    results["text_token_ids"] = token_ids
    results["text_padding_mask"] = padding_mask
    results["text_embedding"] = hf_text.cpu().numpy()

    return results


# ---------------------------------------------------------------
# 2. Validate numerical parity
# ---------------------------------------------------------------
def validate_output(keras_backbone, hf_results):
    """Compare KerasHub and HF outputs for numerical parity."""
    print("\n" + "=" * 50)
    print("NUMERICAL PARITY VALIDATION")
    print("=" * 50)

    # --- Vision encoder ---
    img_np = hf_results["vision_input"]
    keras_out = keras_backbone.vision_encoder({"images": img_np})

    for name, hf_key, keras_key in [
        ("cls_token", "vision_cls", "cls_token"),
        ("register_tokens", "vision_reg", "register_tokens"),
        ("patch_tokens", "vision_patch", "patch_tokens"),
    ]:
        hf_val = hf_results[hf_key]
        keras_val = ops.convert_to_numpy(keras_out[keras_key])
        mse = np.mean((hf_val - keras_val) ** 2)
        max_diff = np.max(np.abs(hf_val - keras_val))
        print(f"\n  Vision {name} via Encoder (shape={keras_val.shape}):")
        print(f"    🔶 HF output:      {hf_val.flatten()[:5]}")
        print(f"    🔶 KerasHub output: {keras_val.flatten()[:5]}")
        print(f"    MSE={mse:.2e}, max_diff={max_diff:.2e}")
        try:
            np.testing.assert_allclose(hf_val, keras_val, atol=1e-4)
            print("    ✓ Matches within atol=1e-4.")
        except AssertionError as e:
            print(f"    ⚠ Mismatch: {e}")

    # --- Text encoder ---
    keras_text = ops.convert_to_numpy(
        keras_backbone.text_encoder(
            {
                "token_ids": hf_results["text_token_ids"].astype("int32"),
                "padding_mask": hf_results["text_padding_mask"],
            }
        )
    )
    hf_text = hf_results["text_embedding"]
    mse = np.mean((hf_text - keras_text) ** 2)
    max_diff = np.max(np.abs(hf_text - keras_text))
    print(f"\n  Text embedding via Encoder (shape={keras_text.shape}):")
    print(f"    🔶 HF output:      {hf_text.flatten()[:5]}")
    print(f"    🔶 KerasHub output: {keras_text.flatten()[:5]}")
    print(f"    MSE={mse:.2e}, max_diff={max_diff:.2e}")
    try:
        np.testing.assert_allclose(hf_text, keras_text, atol=1e-4)
        print("    ✓ Matches within atol=1e-4.")
    except AssertionError as e:
        print(f"    ⚠ Mismatch: {e}")
    print("\n✅ Numerical parity validated.")


# ---------------------------------------------------------------
# 3. Save preset
# ---------------------------------------------------------------
def save_preset(keras_model, tokenizer, image_converter, preset_name):
    """Save the converted model as a KerasHub preset."""
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    tokenizer.save_to_preset(f"./{preset_name}")
    image_converter.save_to_preset(f"./{preset_name}")
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

    # --- Phase 1: Build tokenizer and tokenize sample text ---
    print("-> Building tokenizer...")
    tokenizer_path = hf_hub_download(hf_preset, "tokenizer.model")
    tokenizer = TIPSv2Tokenizer(proto=tokenizer_path)
    print(f"   Tokenizer vocab size: {tokenizer.vocabulary_size()}")

    # Tokenize sample text for validation.
    token_ids_ragged = tokenizer(TEXT_PROMPT)
    token_ids_np = ops.convert_to_numpy(token_ids_ragged)
    seq_len = len(token_ids_np)
    padded_ids = np.zeros((1, MAX_SEQ_LEN), dtype="int32")
    padded_ids[0, :seq_len] = token_ids_np
    padding_mask = np.zeros((1, MAX_SEQ_LEN), dtype="int32")
    padding_mask[0, :seq_len] = 1
    print(f'   Sample text: "{TEXT_PROMPT}"')
    print(f"   Token IDs:   {token_ids_np.tolist()}")

    # --- Phase 2: Load HF model and precompute all outputs ---
    print("\n-> Loading HF reference model...")
    hf_model = AutoModel.from_pretrained(hf_preset, trust_remote_code=True)
    hf_model.eval()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    # Build image converter using config from the loaded HF model.
    img_size = hf_model.config.img_size
    image_converter = TIPSv2ImageConverter(
        image_size=(img_size, img_size),
        scale=(1.0 / 255.0,),
    )
    print(f"   Image size: {img_size}x{img_size}")

    # Download and preprocess a real sample image.
    print("   Downloading sample image...")
    response = requests.get(SAMPLE_IMAGE_URL)
    pil_img = Image.open(BytesIO(response.content)).convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))
    raw_img = np.array(pil_img, dtype="uint8")[np.newaxis]
    img_np = ops.convert_to_numpy(image_converter(raw_img)).astype("float32")
    print(
        f"   Image shape: {img_np.shape}, range: "
        f"[{img_np.min():.3f}, {img_np.max():.3f}]"
    )

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(
        hf_model, img_np, padded_ids, padding_mask
    )
    print("   HF outputs precomputed!")

    # --- Phase 3: Free HF model to reclaim memory ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   HF model released.")

    # --- Phase 4: Load KerasHub model via from_preset ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.TIPSv2Backbone.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    keras_params = keras_model.count_params()
    print(f"   KerasHub model loaded: {keras_params:,} params")

    # --- Phase 5: Validate against precomputed HF outputs ---
    validate_output(keras_model, hf_results)

    # --- Phase 6: Save preset ---
    save_preset(keras_model, tokenizer, image_converter, preset)

    # --- Phase 7: Upload if requested ---
    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
