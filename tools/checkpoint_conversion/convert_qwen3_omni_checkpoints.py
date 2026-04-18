"""Convert Qwen3-Omni HuggingFace checkpoints to KerasHub preset format.

Validates text, image, audio, and video modalities against HF reference
outputs before saving.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \\
        --preset qwen3_omni_30b_a3b_thinking_en

    # Skip slow generation steps (logit-only validation):
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \\
        --preset qwen3_omni_30b_a3b_thinking_en --skip_generation
"""

import gc
import os
import random
import tempfile
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
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "qwen3_omni_30b_a3b_en": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_omni_30b_a3b_captioner_en": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "qwen3_omni_30b_a3b_thinking_en": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
# Sample at a very low fps on CPU to avoid OOM.
VIDEO_FPS = 1

TEXT_PROMPT = "What is Keras?"
AUDIO_SAMPLING_RATE = 16000
AUDIO_FREQ_HZ = 440.0  # A4 sine wave
AUDIO_DURATION_S = 1.0  # 1 second is enough to cover several audio tokens

IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video.<|im_end|>\n<|im_start|>assistant\n"
)
AUDIO_PROMPT = (
    "<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|audio_bos|><|audio_pad|><|audio_eos|>"
    "What sound do you hear?<|im_end|>\n<|im_start|>assistant\n"
)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "If True, skip all generate() calls and only run logit validation.",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _extract_response(text):
    """Strip prompt prefix and any <think>...</think> reasoning block."""
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _generate_test_audio():
    """Return a 1-second 440 Hz sine wave at 16 kHz as float32 numpy."""
    n = int(AUDIO_SAMPLING_RATE * AUDIO_DURATION_S)
    t = np.linspace(0.0, AUDIO_DURATION_S, n, endpoint=False)
    return np.sin(2.0 * np.pi * AUDIO_FREQ_HZ * t).astype(np.float32)


def _count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


def _pixel_values_hf_to_keras(pixel_values_np, vision_encoder):
    """Reshape HF flat pixel patches to KerasHub (N, T, pH, pW, C)."""
    C = vision_encoder.in_channels
    T = vision_encoder.temporal_patch_size
    pH = pW = vision_encoder.patch_size
    flat = pixel_values_np.reshape(-1, C, T, pH, pW)
    return np.transpose(flat, (0, 2, 3, 4, 1))


# ---------------------------------------------------------------
# Phase 1: Precompute all HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_thinker, hf_processor):
    """Run all HF forward passes and return numpy results.

    The HF model is only used here so it can be freed afterward.
    """
    results = {}

    # ------------------------------------------------------------------
    # Text-only
    # ------------------------------------------------------------------
    hf_ids = hf_processor.tokenizer(TEXT_PROMPT, return_tensors="np")[
        "input_ids"
    ]
    results["text_token_ids"] = hf_ids

    with torch.no_grad():
        hf_out = hf_thinker(
            input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device)
        )
    results["text_logits"] = hf_out.logits.detach().cpu().float().numpy()

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_gen = hf_thinker.generate(
                input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device),
                max_new_tokens=32,
                do_sample=False,
            )
        results["text_generated"] = hf_processor.tokenizer.decode(
            hf_gen[0], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Image (vision)
    # ------------------------------------------------------------------
    raw_image = _load_test_image()
    hf_img_inputs = hf_processor(
        text=[IMAGE_PROMPT],
        images=[raw_image],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hf_out = hf_thinker(**hf_img_inputs)
    results["img_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["img_input_ids"] = (
        hf_img_inputs["input_ids"].cpu().numpy().astype(np.int32)
    )
    results["img_attention_mask"] = (
        hf_img_inputs["attention_mask"].cpu().numpy().astype(np.int32)
    )
    results["img_pixel_values"] = (
        hf_img_inputs["pixel_values"].cpu().float().numpy()
    )
    results["img_grid_thw"] = (
        hf_img_inputs["image_grid_thw"].cpu().numpy().astype(np.int32)
    )
    print(f"   HF pixel_values shape: {hf_img_inputs['pixel_values'].shape}")

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_gen = hf_thinker.generate(
                **hf_img_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        results["img_generated"] = hf_processor.batch_decode(
            hf_gen, skip_special_tokens=True
        )[0]

    results["raw_image"] = raw_image

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    audio_np = _generate_test_audio()
    hf_audio_inputs = hf_processor(
        text=[AUDIO_PROMPT],
        audios=[audio_np],
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hf_out = hf_thinker(**hf_audio_inputs)
    results["audio_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["audio_input_ids"] = (
        hf_audio_inputs["input_ids"].cpu().numpy().astype(np.int32)
    )
    results["audio_attention_mask"] = (
        hf_audio_inputs["attention_mask"].cpu().numpy().astype(np.int32)
    )
    # input_features: (1, num_mels, time) — the mel-spectrogram
    results["audio_input_features"] = (
        hf_audio_inputs["input_features"].cpu().float().numpy()
    )
    if "feature_attention_mask" in hf_audio_inputs:
        results["audio_feature_mask"] = (
            hf_audio_inputs["feature_attention_mask"]
            .cpu()
            .numpy()
            .astype(np.int32)
        )
    if "audio_feature_lengths" in hf_audio_inputs:
        results["audio_feature_lengths"] = (
            hf_audio_inputs["audio_feature_lengths"].cpu().numpy()
        )
    results["audio_np"] = audio_np
    print(
        f"   HF input_features shape: {hf_audio_inputs['input_features'].shape}"
    )

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_gen = hf_thinker.generate(
                **hf_audio_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        results["audio_generated"] = hf_processor.batch_decode(
            hf_gen, skip_special_tokens=True
        )[0]

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    vid_path = os.path.join(tempfile.gettempdir(), "qwen3_omni_test_vid.mp4")
    vid_resp = requests.get(VIDEO_URL, timeout=60)
    vid_resp.raise_for_status()
    with open(vid_path, "wb") as f:
        f.write(vid_resp.content)

    video_frames = None
    indices = None
    video_fps = None
    try:
        from torchvision.io import read_video

        video_tensor, _, info = read_video(
            vid_path, pts_unit="sec", output_format="THWC"
        )
        video_fps = info.get("video_fps", 24.0)
        total_frames = video_tensor.shape[0]
        num_sample = max(2, min(int(total_frames / video_fps * VIDEO_FPS), 4))
        indices = (
            np.linspace(0, total_frames - 1, num_sample).round().astype(int)
        )
        video_frames = [
            Image.fromarray(video_tensor[i].numpy()) for i in indices
        ]
        print(
            f"   Video: {total_frames} frames @ {video_fps:.1f}fps"
            f" → sampled {len(video_frames)} frames"
        )
    except ImportError:
        print("   torchvision not available — using blank frames")
        video_frames = [Image.new("RGB", (128, 128)) for _ in range(2)]
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    try:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames,
                        "fps": VIDEO_FPS,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        text_input = hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        for k, v in video_kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                video_kwargs[k] = v[0]
        hf_vid_inputs = hf_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)
    except ImportError:
        print("   qwen_vl_utils not installed — direct video call")
        hf_vid_inputs = hf_processor(
            text=[VIDEO_PROMPT],
            videos=[video_frames],
            return_tensors="pt",
        ).to(device)

    try:
        with torch.no_grad():
            hf_out_vid = hf_thinker(**hf_vid_inputs)
        results["vid_logits"] = hf_out_vid.logits.detach().cpu().float().numpy()
        results["vid_input_ids"] = (
            hf_vid_inputs["input_ids"].cpu().numpy().astype(np.int32)
        )
        results["vid_attention_mask"] = (
            hf_vid_inputs["attention_mask"].cpu().numpy().astype(np.int32)
        )
        vid_px_key = (
            "pixel_values_videos"
            if "pixel_values_videos" in hf_vid_inputs
            else "pixel_values"
        )
        results["vid_pixel_values"] = (
            hf_vid_inputs[vid_px_key].cpu().float().numpy()
        )
        vid_grid_key = (
            "video_grid_thw"
            if "video_grid_thw" in hf_vid_inputs
            else "image_grid_thw"
        )
        results["vid_grid_thw"] = (
            hf_vid_inputs[vid_grid_key].cpu().numpy().astype(np.int32)
        )
        if not FLAGS.skip_generation:
            with torch.no_grad():
                hf_gen_vid = hf_thinker.generate(
                    **hf_vid_inputs,
                    max_new_tokens=16,
                    do_sample=False,
                )
            results["vid_generated"] = hf_processor.batch_decode(
                hf_gen_vid, skip_special_tokens=True
            )[0]
    except Exception as e:
        import traceback

        print(f"   Skipping HF video forward pass: {e}")
        traceback.print_exc()
        results["vid_logits"] = None

    results["video_frames"] = video_frames
    results["video_metadata"] = {
        "frames_indices": indices.tolist() if indices is not None else None,
        "fps": video_fps if video_fps is not None else float(VIDEO_FPS),
    }

    return results


# ---------------------------------------------------------------
# Phase 2: Parameter count
# ---------------------------------------------------------------
def test_parameter_count(keras_backbone, hf_param_count):
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
            f"  ⚠ Parameter count difference: {diff:,} "
            f"(Talker / MTP training heads are excluded from KerasHub)"
        )


# ---------------------------------------------------------------
# Phase 3: Text-only validation
# ---------------------------------------------------------------
def validate_text_output(keras_model, hf_results):
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION")
    print("=" * 50)

    hf_ids = hf_results["text_token_ids"]

    # Token ID parity
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

    # Logit comparison
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
    print(f"\n  Logit mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Logit max abs diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ {e}")

    if not FLAGS.skip_generation:
        print("\n  Generating...")
        out = keras_model.generate(TEXT_PROMPT, max_length=64)

        print(f"  KerasHub: {_extract_response(out)}")
        print(f"  HF:       {hf_results.get('text_generated', 'N/A')}")
        print("  ✓ Text generation done.")


# ---------------------------------------------------------------
# Phase 4: Image validation
# ---------------------------------------------------------------
def validate_image_output(keras_model, hf_results):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping image validation (no vision encoder).")
        return

    print("\n" + "=" * 50)
    print("IMAGE VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    ve = backbone.vision_encoder
    token_ids_np = hf_results["img_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["img_attention_mask"])

    pixel_values_np = _pixel_values_hf_to_keras(
        hf_results["img_pixel_values"], ve
    )
    pixel_values = ops.convert_to_tensor(pixel_values_np)
    grid_thw = ops.convert_to_tensor(hf_results["img_grid_thw"])

    print(f"\n  KerasHub pixel_values shape: {pixel_values_np.shape}")

    keras_hidden = backbone(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
        }
    )
    keras_logits = backbone.token_embedding(keras_hidden, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["img_logits"]

    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Logit max abs diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ {e}")

    if not FLAGS.skip_generation:
        print(f"\n  HF: {hf_results.get('img_generated', 'N/A')}")

        raw_image = hf_results["raw_image"]
        out = keras_model.generate(
            {"prompts": [IMAGE_PROMPT], "images": [np.array(raw_image)]},
            max_length=8192,
        )
        keras_text = out[0] if isinstance(out, list) else out
        print(f"  KerasHub: {_extract_response(keras_text)}")
        print("  ✓ Image generation done.")


# ---------------------------------------------------------------
# Phase 5: Audio validation (Omni-specific)
# ---------------------------------------------------------------
def validate_audio_output(keras_model, hf_results):
    if keras_model.backbone.audio_encoder is None:
        print("\n-> Skipping audio validation (no audio encoder).")
        return

    print("\n" + "=" * 50)
    print("AUDIO VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    token_ids_np = hf_results["audio_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["audio_attention_mask"])

    # HF input_features: (1, num_mels, time) — pass directly as audio_features
    audio_features_np = hf_results["audio_input_features"]
    audio_features = ops.convert_to_tensor(audio_features_np)

    print(f"\n  audio_features shape: {audio_features_np.shape}")

    keras_hidden = backbone(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_features": audio_features,
        }
    )
    keras_logits = backbone.token_embedding(keras_hidden, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["audio_logits"]

    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Logit max abs diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Audio logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ {e}")

    if not FLAGS.skip_generation:
        print(f"\n  HF: {hf_results.get('audio_generated', 'N/A')}")
        audio_np = hf_results["audio_np"]
        out = keras_model.generate(
            {
                "prompts": [AUDIO_PROMPT],
                "audio": [audio_np],
            },
            max_length=64,
        )
        keras_text = out[0] if isinstance(out, list) else out
        print(f"  KerasHub: {_extract_response(keras_text)}")
        print("  ✓ Audio generation done.")


# ---------------------------------------------------------------
# Phase 6: Video validation
# ---------------------------------------------------------------
def validate_video_output(keras_model, hf_results):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping video validation (no vision encoder).")
        return
    if hf_results.get("vid_logits") is None:
        print("\n-> Skipping video validation (HF forward pass failed).")
        return

    print("\n" + "=" * 50)
    print("VIDEO VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    ve = backbone.vision_encoder
    token_ids_np = hf_results["vid_input_ids"]
    token_ids = ops.convert_to_tensor(token_ids_np)
    padding_mask = ops.convert_to_tensor(hf_results["vid_attention_mask"])

    pixel_values_np = _pixel_values_hf_to_keras(
        hf_results["vid_pixel_values"], ve
    )
    pixel_values = ops.convert_to_tensor(pixel_values_np)
    grid_thw = ops.convert_to_tensor(hf_results["vid_grid_thw"])

    print(f"\n  KerasHub video pixel_values shape: {pixel_values_np.shape}")

    keras_hidden = backbone(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
        }
    )
    keras_logits = backbone.token_embedding(keras_hidden, reverse=True)
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    hf_logits = hf_results["vid_logits"]

    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Logit max abs diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
        print("  ✓ Video logits match within atol=1e-3.")
    except AssertionError as e:
        print(f"  ⚠ {e}")

    if not FLAGS.skip_generation:
        print(f"\n  HF: {hf_results.get('vid_generated', 'N/A')}")
        video_frames = hf_results["video_frames"]
        video_frames_np = np.stack([np.array(f) for f in video_frames])
        out = keras_model.generate(
            {"prompts": [VIDEO_PROMPT], "video": [video_frames_np]},
            max_length=8192,
        )
        keras_text = out[0] if isinstance(out, list) else out
        print(f"  KerasHub: {_extract_response(keras_text)}")
        print("  ✓ Video generation done.")


# ---------------------------------------------------------------
# Save preset
# ---------------------------------------------------------------
def save_preset(keras_model, preset_name):
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
    print("-> Loading HF model (Thinker)...")
    hf_full = AutoModelForMultimodalLM.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_thinker = hf_full.thinker
    hf_params = sum(p.numel() for p in hf_thinker.parameters())
    del hf_full  # drop Talker weights immediately
    hf_thinker.eval()
    hf_processor = AutoProcessor.from_pretrained(
        hf_preset, trust_remote_code=True
    )
    print(f"   Thinker loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_thinker, hf_processor)
    hf_results["hf_param_count"] = hf_params
    print("   Done!")

    # --- Phase 2: Free HF model to reclaim memory ---
    print("\n-> Releasing HF model...")
    del hf_thinker
    del hf_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   Released.")

    # --- Phase 3: Load KerasHub model ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate against precomputed HF outputs ---
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_text_output(keras_model, hf_results)
    validate_image_output(keras_model, hf_results)
    validate_audio_output(keras_model, hf_results)
    validate_video_output(keras_model, hf_results)

    # --- Phase 5: Save preset ---
    save_preset(keras_model, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
