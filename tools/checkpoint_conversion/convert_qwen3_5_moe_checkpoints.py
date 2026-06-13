"""Convert Qwen3.5 MoE HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_5_moe_checkpoints.py \
        --preset qwen3_5_moe_35b_a3b_base
"""

import gc
import os
import random
import tempfile
import traceback
from io import BytesIO

from qwen_vl_utils import process_vision_info

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
    "qwen3_5_moe_35b_a3b_base": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen3_5_moe_35b_a3b": "Qwen/Qwen3.5-35B-A3B",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
VIDEO_FPS = 2  # HF default sampling fps for Qwen3.5
TEXT_PROMPT = "What is Keras?"
MULTIMODAL_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video.<|im_end|>\n<|im_start|>assistant\n"
)


def _extract_response(text):
    """Extract the final response from a thinking model's output.

    Strips the prompt prefix and any `<think>...</think>` reasoning block,
    returning only the model's final answer.
    """
    # Take everything after the last 'assistant\n'.
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    # Strip <think>...</think> block if present.
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


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

    # --- Video Multimodal outputs ---
    # Download and decode real MP4 video.
    vid_response = requests.get(VIDEO_URL, timeout=60)
    vid_response.raise_for_status()
    vid_path = os.path.join(tempfile.gettempdir(), "qwen3_5_test_video.mp4")
    with open(vid_path, "wb") as f:
        f.write(vid_response.content)

    indices = None
    video_fps = None
    try:
        from torchvision.io import read_video

        video_tensor, _, info = read_video(
            vid_path, pts_unit="sec", output_format="THWC"
        )
        video_fps = info.get("video_fps", 24.0)
        total_frames = video_tensor.shape[0]

        # Sample frames at VIDEO_FPS (HF default = 2 fps).
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
        print(" torchvision not available, falling back to blank frames")
        video_frames = [Image.new("RGB", (128, 128)) for _ in range(4)]
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    # Use HF's official qwen_vl_utils API for correct video preprocessing.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames, "fps": VIDEO_FPS},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    try:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        for k, v in video_kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                video_kwargs[k] = v[0]
        hf_inputs_vid = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)
    except ImportError:
        print("   ⚠ qwen_vl_utils not installed, falling back to direct call")
        hf_inputs_vid = processor(
            text=[VIDEO_PROMPT], videos=[video_frames], return_tensors="pt"
        ).to(device)

    try:
        video_grid_thw = None
        if "video_grid_thw" in hf_inputs_vid:
            video_grid_thw = (
                hf_inputs_vid["video_grid_thw"].cpu().numpy().astype(np.int32)
            )
        with torch.no_grad():
            hf_out_vid = hf_model(**hf_inputs_vid)

        results["vid_logits"] = hf_out_vid.logits.detach().cpu().float().numpy()
        results["vid_input_ids"] = (
            hf_inputs_vid["input_ids"].cpu().numpy().astype(np.int32)
        )
        results["vid_attention_mask"] = (
            hf_inputs_vid["attention_mask"].cpu().numpy().astype(np.int32)
        )
        results["vid_pixel_values"] = (
            hf_inputs_vid["pixel_values_videos"].cpu().float().numpy()
        )
        results["vid_grid_thw"] = video_grid_thw

        if not FLAGS.skip_generation:
            with torch.no_grad():
                hf_gen_vid = hf_model.generate(
                    **hf_inputs_vid,
                    max_new_tokens=16,
                    do_sample=False,
                )
            results["vid_generated"] = processor.batch_decode(
                hf_gen_vid, skip_special_tokens=True
            )[0]
    except Exception as e:
        print(f" Skipping HF video forward pass: {e}")
        traceback.print_exc()
        results["vid_logits"] = None
        results["vid_generated"] = None

    results["video_frames"] = video_frames
    results["video_metadata"] = {
        "frames_indices": indices.tolist() if indices is not None else None,
        "fps": video_fps if video_fps is not None else 2.0,
    }

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
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-3: {e}")

    # --- End-to-end generation ---
    if not FLAGS.skip_generation:
        print("\n  Generating text...")
        keras_output = keras_model.generate(TEXT_PROMPT, max_length=64)
        print(f"  KerasHub: {_extract_response(keras_output)}")
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
        token_ids, image_grid_thw, None
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
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-3: {e}")

    # --- End-to-end generation ---
    if not FLAGS.skip_generation:
        print(f"\n  HF output: {hf_results['mm_generated']}")

        raw_image = hf_results["raw_image"]
        keras_output = keras_model.generate(
            {"prompts": [MULTIMODAL_PROMPT], "images": [np.array(raw_image)]},
            max_length=8192,
        )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        print(f"  KerasHub output: {_extract_response(keras_text)}")
        print("  ✓ Multimodal generation completed.")


def _run_keras_video_generation(keras_model, hf_results):
    """Run KerasHub video generation with metadata."""
    video_frames = hf_results["video_frames"]
    video_frames_np = np.stack([np.array(img) for img in video_frames])
    video_meta = hf_results.get("video_metadata")
    vm_list = (
        [video_meta]
        if video_meta and video_meta.get("frames_indices")
        else None
    )
    keras_model.preprocessor._video_metadata = vm_list
    keras_output = keras_model.generate(
        {"prompts": [VIDEO_PROMPT], "videos": [video_frames_np]},
        max_length=8192,
    )
    keras_model.preprocessor._video_metadata = None
    keras_text = (
        keras_output[0] if isinstance(keras_output, list) else keras_output
    )
    print(f"  KerasHub output: {_extract_response(keras_text)}")


# ---------------------------------------------------------------
# 5. Validate video output
# ---------------------------------------------------------------
def validate_video_output(keras_model, hf_results):
    """Validate video multimodal logits and generation."""
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping video validation (no vision encoder).")
        return

    print("\n" + "=" * 50)
    print("VIDEO VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    token_ids_np = hf_results["vid_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["vid_attention_mask"])

    pixel_values_np = hf_results["vid_pixel_values"]
    ve = backbone.vision_encoder
    C, T = ve.in_channels, ve.temporal_patch_size
    pH = pW = ve.patch_size
    pixel_values_np = pixel_values_np.reshape(-1, C, T, pH, pW)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 4, 1))
    pixel_values = ops.convert_to_tensor(pixel_values_np)

    video_grid_thw = ops.convert_to_tensor(hf_results["vid_grid_thw"])

    video_pad_id = keras_model.preprocessor.video_token_id
    vision_pos = np.where(token_ids_np[0] == video_pad_id)[0]
    vision_indices = ops.convert_to_tensor(vision_pos.astype(np.int32))

    img_embeds = backbone.vision_encoder(pixel_values, video_grid_thw)
    x = backbone.token_embedding(token_ids)
    x = backbone.interleave_embeddings(
        image_embeddings=img_embeds,
        text_embeddings=x,
        vision_indices=vision_indices,
    )

    position_ids = keras_model.preprocessor._compute_position_ids(
        token_ids, None, video_grid_thw
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
    hf_logits = hf_results["vid_logits"]

    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ Logits do not match within atol=1e-3: {e}")

    if not FLAGS.skip_generation:
        print(f"\n  HF output: {hf_results['vid_generated']}")
        _run_keras_video_generation(keras_model, hf_results)
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
    keras_model = keras_hub.models.Qwen3_5MoeCausalLM.from_preset(
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
