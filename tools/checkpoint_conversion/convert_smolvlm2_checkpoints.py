"""Convert SmolVLM2 HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_smolvlm2_checkpoints.py \
        --preset smolvlm2_256m_video_instruct
"""

import gc
import os
import random
import tempfile

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from io import BytesIO

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
    "smolvlm2_256m_video_instruct": (
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    ),
    "smolvlm2_500m_video_instruct": (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ),
    "smolvlm2_2.2b_instruct": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
VIDEO_FPS = 1  # SmolVLM2 default sampling fps
TEXT_PROMPT = "What is Keras?"
MULTIMODAL_TEXT = "Describe this image in detail."
VIDEO_TEXT = "Describe what is happening in this video."
# KerasHub prompt with <image> placeholder and chat formatting.
KERASHUB_MULTIMODAL_PROMPT = (
    "<|im_start|>User:<image>"
    + MULTIMODAL_TEXT
    + "<end_of_utterance>\nAssistant:"
)
KERASHUB_VIDEO_PROMPT = (
    "<|im_start|>User:<video>" + VIDEO_TEXT + "<end_of_utterance>\nAssistant:"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "If True, skip all text generation steps and only run "
    "numerical logit validation.",
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

    if not FLAGS.skip_generation:
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

    # Build chat-style prompt with image.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": MULTIMODAL_TEXT},
            ],
        }
    ]
    mm_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    hf_inputs = processor(
        text=mm_prompt, images=[raw_image], return_tensors="pt"
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

    if not FLAGS.skip_generation:
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

    # --- Video outputs ---
    # Download and decode real MP4 video.
    print("\n   Downloading test video...")
    vid_response = requests.get(VIDEO_URL, timeout=60)
    vid_response.raise_for_status()
    vid_path = os.path.join(tempfile.gettempdir(), "smolvlm2_test_video.mp4")
    with open(vid_path, "wb") as f:
        f.write(vid_response.content)

    video_fps = None
    try:
        from torchvision.io import read_video

        video_tensor, _, info = read_video(
            vid_path, pts_unit="sec", output_format="THWC"
        )
        video_fps = info.get("video_fps", 24.0)
        total_frames = video_tensor.shape[0]

        # Sample frames at VIDEO_FPS (SmolVLM2 default = 1 fps).
        # Cap at 8 to avoid OOM on limited-memory devices.
        num_sample = max(4, min(int(total_frames / video_fps * VIDEO_FPS), 8))
        indices = (
            np.linspace(0, total_frames - 1, num_sample).round().astype(int)
        )
        sampled_frames = video_tensor[indices]  # (N, H, W, 3) uint8

        # Convert to list of PIL images (what HF processor expects).
        video_frames = [
            Image.fromarray(sampled_frames[i].numpy())
            for i in range(sampled_frames.shape[0])
        ]
        print(
            f"   Video: {total_frames} total frames @ {video_fps}fps ->"
            f" sampled {len(video_frames)} frames"
        )
    except ImportError:
        print("   ⚠ torchvision not available, falling back to blank frames")
        video_frames = [Image.new("RGB", (128, 128)) for _ in range(4)]
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    # Build video chat prompt via HF.
    video_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": VIDEO_TEXT},
            ],
        }
    ]
    video_prompt = processor.apply_chat_template(
        video_messages, add_generation_prompt=True
    )
    video_inputs = processor(
        text=video_prompt, videos=[[video_frames]], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        hf_video_out = hf_model(**video_inputs)
    results["video_logits"] = hf_video_out.logits.detach().cpu().float().numpy()
    print(
        f"   HF video pixel_values shape: {video_inputs['pixel_values'].shape}"
    )

    results["video_input_ids"] = (
        video_inputs["input_ids"].cpu().numpy().astype(np.int32)
    )
    results["video_attention_mask"] = (
        video_inputs["attention_mask"].cpu().numpy().astype(np.int32)
    )
    results["video_pixel_values"] = (
        video_inputs["pixel_values"].cpu().float().numpy()
    )

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_video_gen = hf_model.generate(
                **video_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        results["video_generated"] = processor.batch_decode(
            hf_video_gen, skip_special_tokens=True
        )[0]
    results["video_frames"] = video_frames

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
        print(f"  ⚠ Parameter count difference: {diff:,}")


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
    try:
        np.testing.assert_array_equal(keras_valid, hf_ids[0])
        print("  ✓ Token IDs match.")
    except AssertionError as e:
        print(f"  ⚠ Token IDs mismatch: {e}")

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
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-3: {e}")

    if not FLAGS.skip_generation:
        # --- End-to-end generation ---
        print("\n  Generating text...")
        keras_output = keras_model.generate(TEXT_PROMPT, max_length=64)
        print(f"  KerasHub: {keras_output}")
        print(f"  HF:       {hf_results.get('text_generated', 'N/A')}")
        print("  ✓ Text generation completed.")


# ---------------------------------------------------------------
# 4. Validate multimodal output
# ---------------------------------------------------------------
def validate_multimodal_output(keras_model, hf_results):
    """Validate multimodal logits and generation."""
    print("\n" + "=" * 50)
    print("MULTIMODAL VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    token_ids_np = hf_results["mm_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["mm_attention_mask"])

    # Process images through the vision pipeline.
    # HF pixel_values shape: (batch, num_images, C, H, W)
    # → KerasHub expects (batch * num_images, H, W, C)
    pixel_values_np = hf_results["mm_pixel_values"]
    if pixel_values_np.ndim == 5:
        b, n, c, h, w = pixel_values_np.shape
        pixel_values_np = pixel_values_np.reshape(b * n, c, h, w)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 1))
    pixel_values = ops.convert_to_tensor(pixel_values_np)

    # Encode images through vision encoder + connector.
    img_embeds = backbone.vision_encoder({"pixel_values": pixel_values})
    img_embeds = backbone.connector(img_embeds)

    # Compute vision_indices from image_token_id positions.
    vision_pos = np.where(token_ids_np[0] == backbone.image_token_id)[0]
    vision_indices = ops.convert_to_tensor(
        vision_pos.astype(np.int32)[np.newaxis, :]
    )

    # Get text embeddings and merge with vision embeddings.
    text_embeddings = backbone.token_embedding(token_ids)
    merged = backbone.interleave_embeddings(
        image_embeddings=img_embeds,
        text_embeddings=text_embeddings,
        vision_indices=vision_indices,
    )

    # Forward through decoder layers.
    x = merged
    for layer in backbone.transformer_layers:
        x = layer(x, decoder_padding_mask=padding_mask)

    x = backbone.layer_norm(x)
    keras_logits = backbone.token_embedding(x, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["mm_logits"]

    # --- Logit comparison ---
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Multimodal logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Multimodal logits do not match within atol=1e-3: {e}")

    if not FLAGS.skip_generation:
        # --- End-to-end generation ---
        print(f"\n  HF output: {hf_results.get('mm_generated', 'N/A')}")

        raw_image = hf_results["raw_image"]
        keras_output = keras_model.generate(
            {
                "prompts": [KERASHUB_MULTIMODAL_PROMPT],
                "images": [np.array(raw_image)],
            },
            max_length=1024,
        )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        print(f"  KerasHub output: {keras_text}")
        print("  ✓ Multimodal generation completed.")


# ---------------------------------------------------------------
# 5. Validate video output
# ---------------------------------------------------------------
def validate_video_output(keras_model, hf_results):
    """Validate video logits and generation."""
    print("\n" + "=" * 50)
    print("VIDEO VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    token_ids_np = hf_results["video_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["video_attention_mask"])

    # Process video frames through the vision pipeline.
    # HF video pixel_values shape: (batch, num_frames, C, H, W)
    # → KerasHub expects (num_frames, H, W, C)
    pixel_values_np = hf_results["video_pixel_values"]
    if pixel_values_np.ndim == 5:
        b, n, c, h, w = pixel_values_np.shape
        pixel_values_np = pixel_values_np.reshape(b * n, c, h, w)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 1))
    pixel_values = ops.convert_to_tensor(pixel_values_np)

    print(f"\n  KerasHub video pixel_values shape: {pixel_values.shape}")

    # Encode frames through vision encoder + connector.
    img_embeds = backbone.vision_encoder({"pixel_values": pixel_values})
    img_embeds = backbone.connector(img_embeds)

    # Compute vision_indices from image_token_id positions.
    vision_pos = np.where(token_ids_np[0] == backbone.image_token_id)[0]
    vision_indices = ops.convert_to_tensor(
        vision_pos.astype(np.int32)[np.newaxis, :]
    )

    print(f"  Video vision_indices count: {len(vision_pos)}")

    # Get text embeddings and merge with vision embeddings.
    text_embeddings = backbone.token_embedding(token_ids)
    merged = backbone.interleave_embeddings(
        image_embeddings=img_embeds,
        text_embeddings=text_embeddings,
        vision_indices=vision_indices,
    )

    # Forward through decoder layers.
    x = merged
    for layer in backbone.transformer_layers:
        x = layer(x, decoder_padding_mask=padding_mask)

    x = backbone.layer_norm(x)
    keras_logits = backbone.token_embedding(x, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["video_logits"]

    # --- Logit comparison ---
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Video logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Video logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Video logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Video logits do not match within atol=1e-3: {e}")

    if not FLAGS.skip_generation:
        # --- End-to-end video generation ---
        print(
            f"\n  HF video output: {hf_results.get('video_generated', 'N/A')}"
        )

        # Build video tensor from PIL frames.
        video_frames = hf_results["video_frames"]
        video_np = np.stack(
            [np.array(f) for f in video_frames], axis=0
        )  # (num_frames, H, W, 3)

        keras_output = keras_model.generate(
            {
                "prompts": [KERASHUB_VIDEO_PROMPT],
                "videos": [video_np],
            },
            max_length=1024,
        )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        print(f"  KerasHub video output: {keras_text}")
        print("  ✓ Video generation completed.")


# ---------------------------------------------------------------
# 6. Save preset
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
    keras_model = keras_hub.models.SmolVLM2CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate against precomputed HF outputs ---
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_text_output(keras_model, hf_results)
    validate_multimodal_output(keras_model, hf_results)
    validate_video_output(keras_model, hf_results)

    # --- Phase 5: Save preset ---
    save_preset(keras_model, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
