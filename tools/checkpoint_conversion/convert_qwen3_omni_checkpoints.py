"""Convert Qwen3-Omni HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \\
        --preset qwen3_omni_30b_a3b_thinking_en

    # Halve peak RAM (runs HF and Keras in bfloat16):
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \\
        --preset qwen3_omni_30b_a3b_thinking_en --dtype bfloat16

    # Skip slow generation steps (logit-only validation):
    python tools/checkpoint_conversion/convert_qwen3_omni_checkpoints.py \\
        --preset qwen3_omni_30b_a3b_thinking_en --skip_generation
"""

import gc
import json
import os
import random
import shutil
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

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/"
    "video1.mp4"
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
flags.DEFINE_enum(
    "dtype",
    "float32",
    list(DTYPE_MAP.keys()),
    "Precision for both HF and Keras models. bfloat16 halves peak RAM "
    "vs float32 at the cost of a small numerical tolerance in the logit "
    "comparison.",
)
flags.DEFINE_string(
    "cache_dir",
    None,
    "Directory to persist HF reference outputs to. If unset, a temp dir "
    "is created and deleted on exit. Set this to a durable path to enable "
    "resuming Phase B (Keras validation) without rerunning Phase A (HF "
    "forward passes).",
)
flags.DEFINE_bool(
    "skip_hf",
    False,
    "Skip Phase A entirely and reuse the HF outputs already present in "
    "--cache_dir. Use this to iterate on Keras-side validation without "
    "redoing slow HF forward passes.",
)
flags.DEFINE_bool("skip_text", False, "Skip text validation.")
flags.DEFINE_bool("skip_image", False, "Skip image validation.")
flags.DEFINE_bool("skip_audio", False, "Skip audio validation.")
flags.DEFINE_bool("skip_video", False, "Skip video validation.")
flags.DEFINE_bool(
    "skip_save",
    False,
    "Skip saving the KerasHub preset to disk (Phase C).",
)
flags.DEFINE_bool(
    "keep_cache",
    False,
    "If True, do not delete the cache_dir on exit (implied when "
    "--cache_dir is set by the user).",
)


# ---------------------------------------------------------------
# Generic helpers
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


def _logit_tolerance(dtype_str):
    """Loose atol for bfloat16, tight for float32."""
    return 5e-2 if dtype_str == "bfloat16" else 1e-3


def _atol_str(dtype_str):
    return f"{_logit_tolerance(dtype_str):.0e}"


def _cast_inputs_to_model_dtype(inputs, model):
    """Cast float tensors in a processor dict to the model's parameter dtype."""
    model_dtype = next(model.parameters()).dtype
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model_dtype)
    return inputs


def _freeze_keras_model(keras_model):
    """Disable gradient tracking on every Keras variable.

    Without this, Keras's torch backend leaves `requires_grad=True` on all
    ~30 B parameters. Every forward pass through the MoE decoder then builds
    a full autograd graph.
    """
    keras_model.trainable = False
    for var in keras_model.variables:
        t = getattr(var, "value", var)
        if hasattr(t, "requires_grad_"):
            t.requires_grad_(False)


def _inference_mode():
    """Context manager that disables autograd for the wrapped Keras call."""
    return torch.inference_mode()


# ---------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------
# Each modality is persisted as <modality>.npz (numpy arrays) +
# <modality>.json (string metadata, e.g. generated text). Raw inputs
# needed to re-run Keras generate() (raw PNG image, audio .npy, video
# frame PNGs) are also written here.


def _save_arrays(cache_dir, name, arrays, meta=None):
    path = os.path.join(cache_dir, f"{name}.npz")
    np.savez(path, **arrays)
    if meta is not None:
        with open(os.path.join(cache_dir, f"{name}.json"), "w") as f:
            json.dump(meta, f)


def _load_arrays(cache_dir, name):
    with np.load(os.path.join(cache_dir, f"{name}.npz")) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta_path = os.path.join(cache_dir, f"{name}.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return arrays, meta


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------
# Phase A: HF output computation (one modality at a time, persisted
# to disk, freed immediately)
# ---------------------------------------------------------------


def _hf_text(hf_thinker, hf_processor, cache_dir):
    print("\n-> HF text...")
    ids = hf_processor.tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    with torch.no_grad():
        out = hf_thinker(
            input_ids=torch.tensor(ids, dtype=torch.long).to(device)
        )
    logits_np = out.logits.detach().cpu().float().numpy()
    del out
    _free()

    meta = {}
    if not FLAGS.skip_generation:
        with torch.no_grad():
            gen = hf_thinker.generate(
                input_ids=torch.tensor(ids, dtype=torch.long).to(device),
                max_new_tokens=32,
                do_sample=False,
            )
        meta["generated"] = hf_processor.tokenizer.decode(
            gen[0], skip_special_tokens=True
        )
        del gen
        _free()

    _save_arrays(
        cache_dir,
        "text",
        {"token_ids": ids.astype(np.int32), "logits": logits_np},
        meta,
    )
    del logits_np
    _free()


def _hf_image(hf_thinker, hf_processor, cache_dir):
    print("\n-> HF image...")
    raw = _load_test_image()
    # Save raw image for Keras generate() reload later.
    raw.save(os.path.join(cache_dir, "image.png"))

    inputs = hf_processor(
        text=[IMAGE_PROMPT],
        images=[raw],
        return_tensors="pt",
    ).to(device)
    _cast_inputs_to_model_dtype(inputs, hf_thinker)
    print(f"   HF pixel_values shape: {inputs['pixel_values'].shape}")
    del raw

    with torch.no_grad():
        out = hf_thinker(**inputs)
    arrays = {
        "logits": out.logits.detach().cpu().float().numpy(),
        "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int32),
        "attention_mask": (
            inputs["attention_mask"].cpu().numpy().astype(np.int32)
        ),
        "pixel_values": inputs["pixel_values"].cpu().float().numpy(),
        "grid_thw": (inputs["image_grid_thw"].cpu().numpy().astype(np.int32)),
    }
    del out
    _free()

    meta = {}
    if not FLAGS.skip_generation:
        with torch.no_grad():
            gen = hf_thinker.generate(
                **inputs, max_new_tokens=32, do_sample=False
            )
        meta["generated"] = hf_processor.batch_decode(
            gen, skip_special_tokens=True
        )[0]
        del gen
        _free()

    _save_arrays(cache_dir, "image", arrays, meta)
    del inputs, arrays
    _free()


def _hf_audio(hf_thinker, hf_processor, cache_dir):
    print("\n-> HF audio...")
    audio_np = _generate_test_audio()
    np.save(os.path.join(cache_dir, "audio.npy"), audio_np)

    inputs = hf_processor(
        text=[AUDIO_PROMPT],
        audio=[audio_np],
        return_tensors="pt",
        padding=True,
    ).to(device)
    _cast_inputs_to_model_dtype(inputs, hf_thinker)
    print(f"   HF input_features shape: {inputs['input_features'].shape}")
    del audio_np

    with torch.no_grad():
        out = hf_thinker(**inputs)
    arrays = {
        "logits": out.logits.detach().cpu().float().numpy(),
        "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int32),
        "attention_mask": (
            inputs["attention_mask"].cpu().numpy().astype(np.int32)
        ),
        # input_features: (1, num_mels, time) — the mel-spectrogram.
        "input_features": inputs["input_features"].cpu().float().numpy(),
    }
    del out
    _free()

    meta = {}
    if not FLAGS.skip_generation:
        with torch.no_grad():
            gen = hf_thinker.generate(
                **inputs, max_new_tokens=32, do_sample=False
            )
        meta["generated"] = hf_processor.batch_decode(
            gen, skip_special_tokens=True
        )[0]
        del gen
        _free()

    _save_arrays(cache_dir, "audio", arrays, meta)
    del inputs, arrays
    _free()


def _hf_video(hf_thinker, hf_processor, cache_dir):
    print("\n-> HF video...")
    vid_path = os.path.join(tempfile.gettempdir(), "qwen3_omni_test_vid.mp4")
    vid_resp = requests.get(VIDEO_URL, timeout=60)
    vid_resp.raise_for_status()
    with open(vid_path, "wb") as f:
        f.write(vid_resp.content)

    video_frames = None
    video_fps = None
    indices = None
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
        del video_tensor
    except ImportError:
        print("   torchvision not available — using blank frames")
        video_frames = [Image.new("RGB", (128, 128)) for _ in range(2)]
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    # Persist frames as PNGs so Keras generate() can reload them.
    frames_dir = os.path.join(cache_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frame_names = []
    for i, frame in enumerate(video_frames):
        name = f"frame_{i:04d}.png"
        frame.save(os.path.join(frames_dir, name))
        frame_names.append(name)

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
        inputs = hf_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)
    except ImportError:
        print("   qwen_vl_utils not installed — direct video call")
        inputs = hf_processor(
            text=[VIDEO_PROMPT],
            videos=[video_frames],
            return_tensors="pt",
        ).to(device)
    _cast_inputs_to_model_dtype(inputs, hf_thinker)

    del video_frames
    _free()

    meta = {"frames": frame_names, "fps": float(VIDEO_FPS)}

    try:
        with torch.no_grad():
            out = hf_thinker(**inputs)
        vid_px_key = (
            "pixel_values_videos"
            if "pixel_values_videos" in inputs
            else "pixel_values"
        )
        vid_grid_key = (
            "video_grid_thw" if "video_grid_thw" in inputs else "image_grid_thw"
        )
        arrays = {
            "logits": out.logits.detach().cpu().float().numpy(),
            "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int32),
            "attention_mask": (
                inputs["attention_mask"].cpu().numpy().astype(np.int32)
            ),
            "pixel_values": inputs[vid_px_key].cpu().float().numpy(),
            "grid_thw": inputs[vid_grid_key].cpu().numpy().astype(np.int32),
        }
        del out
        _free()

        if not FLAGS.skip_generation:
            with torch.no_grad():
                gen = hf_thinker.generate(
                    **inputs, max_new_tokens=16, do_sample=False
                )
            meta["generated"] = hf_processor.batch_decode(
                gen, skip_special_tokens=True
            )[0]
            del gen
            _free()

        _save_arrays(cache_dir, "video", arrays, meta)
        del arrays
    except Exception as e:
        import traceback

        print(f"   Skipping HF video forward pass: {e}")
        traceback.print_exc()
        # Sentinel: write only meta so Keras phase can skip cleanly.
        meta["skipped"] = True
        with open(os.path.join(cache_dir, "video.json"), "w") as f:
            json.dump(meta, f)

    del inputs
    _free()


# ---------------------------------------------------------------
# Phase B: KerasHub validation (one modality at a time, loaded from
# disk, freed after comparison)
# ---------------------------------------------------------------


def _print_logit_diff(keras_logits, hf_logits, label, atol):
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  {label} logit mean abs diff: {abs_diff.mean():.6f}")
    print(f"  {label} logit max abs diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=atol)
        print(f"  ✓ {label} logits match within atol={atol:.0e}.")
    except AssertionError as e:
        print(f"  ⚠ {e}")


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


def validate_text_output(keras_model, cache_dir, dtype_str):
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION")
    print("=" * 50)

    arrays, meta = _load_arrays(cache_dir, "text")
    hf_ids = arrays["token_ids"]
    hf_logits = arrays["logits"]

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

    token_ids = ops.convert_to_tensor(hf_ids.astype(np.int32))
    padding_mask = ops.ones_like(token_ids)
    with _inference_mode():
        keras_hidden = keras_model.backbone(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        keras_logits = keras_model.backbone.token_embedding(
            keras_hidden, reverse=True
        )
        keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    del keras_hidden

    _print_logit_diff(
        keras_logits, hf_logits, "Text", _logit_tolerance(dtype_str)
    )
    del keras_logits, hf_logits, arrays
    _free()

    if not FLAGS.skip_generation:
        print("\n  Generating...")
        with _inference_mode():
            out = keras_model.generate(TEXT_PROMPT, max_length=64)
        print(f"  KerasHub: {_extract_response(out)}")
        print(f"  HF:       {meta.get('generated', 'N/A')}")
        print("  ✓ Text generation done.")
        _free()


def validate_image_output(keras_model, cache_dir, dtype_str):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping image validation (no vision encoder).")
        return
    if not os.path.exists(os.path.join(cache_dir, "image.npz")):
        print("\n-> Skipping image validation (no cached HF output).")
        return

    print("\n" + "=" * 50)
    print("IMAGE VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    ve = backbone.vision_encoder
    arrays, meta = _load_arrays(cache_dir, "image")

    token_ids = ops.convert_to_tensor(arrays["input_ids"])
    padding_mask = ops.convert_to_tensor(arrays["attention_mask"])
    pixel_values_np = _pixel_values_hf_to_keras(arrays["pixel_values"], ve)
    pixel_values = ops.convert_to_tensor(pixel_values_np)
    grid_thw = ops.convert_to_tensor(arrays["grid_thw"])
    hf_logits = arrays["logits"]
    print(f"\n  KerasHub pixel_values shape: {pixel_values_np.shape}")
    del pixel_values_np

    with _inference_mode():
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
    del keras_hidden, token_ids, padding_mask, pixel_values, grid_thw

    _print_logit_diff(
        keras_logits, hf_logits, "Image", _logit_tolerance(dtype_str)
    )
    prompt_len = int(arrays["input_ids"].shape[1])
    del keras_logits, hf_logits, arrays
    _free()

    if not FLAGS.skip_generation:
        print(f"\n  HF: {meta.get('generated', 'N/A')}")
        raw_image = np.array(
            Image.open(os.path.join(cache_dir, "image.png")).convert("RGB")
        )
        with _inference_mode():
            out = keras_model.generate(
                {"prompts": [IMAGE_PROMPT], "images": [raw_image]},
                max_length=prompt_len + 32,
            )
        keras_text = out[0] if isinstance(out, list) else out
        print(f"  KerasHub: {_extract_response(keras_text)}")
        print("  ✓ Image generation done.")
        del raw_image
        _free()


def validate_audio_output(keras_model, cache_dir, dtype_str):
    if keras_model.backbone.audio_encoder is None:
        print("\n-> Skipping audio validation (no audio encoder).")
        return
    if not os.path.exists(os.path.join(cache_dir, "audio.npz")):
        print("\n-> Skipping audio validation (no cached HF output).")
        return

    print("\n" + "=" * 50)
    print("AUDIO VALIDATION")
    print("=" * 50)

    arrays, meta = _load_arrays(cache_dir, "audio")
    token_ids = ops.convert_to_tensor(arrays["input_ids"])
    padding_mask = ops.convert_to_tensor(arrays["attention_mask"])
    audio_features = ops.convert_to_tensor(arrays["input_features"])
    hf_logits = arrays["logits"]
    print(f"\n  audio_features shape: {arrays['input_features'].shape}")

    with _inference_mode():
        keras_hidden = keras_model.backbone(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "audio_features": audio_features,
            }
        )
        keras_logits = keras_model.backbone.token_embedding(
            keras_hidden, reverse=True
        )
        keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)
    del keras_hidden, token_ids, padding_mask, audio_features

    _print_logit_diff(
        keras_logits, hf_logits, "Audio", _logit_tolerance(dtype_str)
    )
    prompt_len = int(arrays["input_ids"].shape[1])
    del keras_logits, hf_logits, arrays
    _free()

    if not FLAGS.skip_generation:
        print(f"\n  HF: {meta.get('generated', 'N/A')}")
        audio_np = np.load(os.path.join(cache_dir, "audio.npy"))
        with _inference_mode():
            out = keras_model.generate(
                {"prompts": [AUDIO_PROMPT], "audio": [audio_np]},
                max_length=prompt_len + 32,
            )
        keras_text = out[0] if isinstance(out, list) else out
        print(f"  KerasHub: {_extract_response(keras_text)}")
        print("  ✓ Audio generation done.")
        del audio_np
        _free()


def validate_video_output(keras_model, cache_dir, dtype_str):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping video validation (no vision encoder).")
        return
    if not os.path.exists(os.path.join(cache_dir, "video.npz")):
        meta_path = os.path.join(cache_dir, "video.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("skipped"):
                print(
                    "\n-> Skipping video validation (HF forward pass failed)."
                )
                return
        print("\n-> Skipping video validation (no cached HF output).")
        return

    print("\n" + "=" * 50)
    print("VIDEO VALIDATION")
    print("=" * 50)

    backbone = keras_model.backbone
    ve = backbone.vision_encoder
    arrays, meta = _load_arrays(cache_dir, "video")

    token_ids = ops.convert_to_tensor(arrays["input_ids"])
    padding_mask = ops.convert_to_tensor(arrays["attention_mask"])
    pixel_values_np = _pixel_values_hf_to_keras(arrays["pixel_values"], ve)
    pixel_values = ops.convert_to_tensor(pixel_values_np)
    grid_thw = ops.convert_to_tensor(arrays["grid_thw"])
    hf_logits = arrays["logits"]
    print(f"\n  KerasHub video pixel_values shape: {pixel_values_np.shape}")
    del pixel_values_np

    with _inference_mode():
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
    del keras_hidden, token_ids, padding_mask, pixel_values, grid_thw

    _print_logit_diff(
        keras_logits, hf_logits, "Video", _logit_tolerance(dtype_str)
    )
    prompt_len = int(arrays["input_ids"].shape[1])
    del keras_logits, hf_logits, arrays
    _free()

    if not FLAGS.skip_generation:
        print(f"\n  HF: {meta.get('generated', 'N/A')}")
        frames_dir = os.path.join(cache_dir, "video_frames")
        frame_names = meta.get("frames", [])
        frames = [
            np.array(Image.open(os.path.join(frames_dir, n)).convert("RGB"))
            for n in frame_names
        ]
        if frames:
            video_frames_np = np.stack(frames)
            with _inference_mode():
                out = keras_model.generate(
                    {"prompts": [VIDEO_PROMPT], "video": [video_frames_np]},
                    max_length=prompt_len + 32,
                )
            keras_text = out[0] if isinstance(out, list) else out
            print(f"  KerasHub: {_extract_response(keras_text)}")
            print("  ✓ Video generation done.")
            del video_frames_np
        del frames
        _free()


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
    dtype_str = FLAGS.dtype
    torch_dtype = DTYPE_MAP[dtype_str]

    user_cache = FLAGS.cache_dir
    if user_cache:
        cache_dir = os.path.abspath(user_cache)
        os.makedirs(cache_dir, exist_ok=True)
        keep_cache = True  # never auto-delete a path the user chose
    else:
        cache_dir = tempfile.mkdtemp(prefix="qwen3_omni_xfer_")
        keep_cache = FLAGS.keep_cache
    print(f"-> Cache dir: {cache_dir}  (keep={keep_cache})")
    print(f"-> dtype:     {dtype_str}")

    try:
        # =================================================================
        # Phase A: Load HF, compute + persist outputs per modality, free HF.
        # =================================================================
        if FLAGS.skip_hf:
            meta_path = os.path.join(cache_dir, "meta.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"--skip_hf was set but {meta_path} does not exist. "
                    "Run Phase A first (without --skip_hf) to populate the "
                    "cache."
                )
            print("\n-> Skipping Phase A (using cached HF outputs).")
        else:
            print("\n-> Loading HF model (Thinker)...")
            hf_full = AutoModelForMultimodalLM.from_pretrained(
                hf_preset,
                device_map="cpu",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            hf_thinker = hf_full.thinker
            hf_params = sum(p.numel() for p in hf_thinker.parameters())
            del hf_full  # drop Talker weights immediately
            _free()
            hf_thinker.eval()
            hf_processor = AutoProcessor.from_pretrained(
                hf_preset, trust_remote_code=True
            )
            print(f"   Thinker loaded: {hf_params:,} params")

            _hf_text(hf_thinker, hf_processor, cache_dir)
            _hf_image(hf_thinker, hf_processor, cache_dir)
            _hf_audio(hf_thinker, hf_processor, cache_dir)
            _hf_video(hf_thinker, hf_processor, cache_dir)

            with open(os.path.join(cache_dir, "meta.json"), "w") as f:
                json.dump({"hf_params": hf_params, "dtype": dtype_str}, f)

            print("\n-> Releasing HF model...")
            del hf_thinker
            del hf_processor
            _free()
            print("   Released.")

        # =================================================================
        # Phase B: Load KerasHub, validate each modality from disk.
        # =================================================================
        print("\n-> Loading KerasHub model from HF preset...")
        keras_model = keras_hub.models.Qwen3OmniCausalLM.from_preset(
            f"hf://{hf_preset}", dtype=dtype_str
        )
        _freeze_keras_model(keras_model)
        _free()
        print("   KerasHub model loaded (autograd disabled, params frozen)!")

        with open(os.path.join(cache_dir, "meta.json")) as f:
            run_meta = json.load(f)

        test_parameter_count(keras_model.backbone, run_meta["hf_params"])

        if FLAGS.skip_text:
            print("\n-> Skipping text validation.")
        else:
            validate_text_output(keras_model, cache_dir, dtype_str)
        if FLAGS.skip_image:
            print("\n-> Skipping image validation.")
        else:
            validate_image_output(keras_model, cache_dir, dtype_str)
        if FLAGS.skip_audio:
            print("\n-> Skipping audio validation.")
        else:
            validate_audio_output(keras_model, cache_dir, dtype_str)
        if FLAGS.skip_video:
            print("\n-> Skipping video validation.")
        else:
            validate_video_output(keras_model, cache_dir, dtype_str)

        # =================================================================
        # Phase C: Save preset.
        # =================================================================
        if FLAGS.skip_save:
            print("\n-> Skipping preset save (--skip_save).")
        else:
            save_preset(keras_model, preset)

        print("\n=== Done! ===")
    finally:
        if not keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            print(f"\n-> Cache retained at: {cache_dir}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
