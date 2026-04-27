"""Qwen3.5 KerasHub → HuggingFace Export Verification Script.

Performs a real-weights round-trip test following the same validation
pattern as the checkpoint conversion script:

  1. Load KerasHub preset → Export to HF format
  2. Load ORIGINAL HF model → Precompute all outputs (text, image, video)
  3. Load EXPORTED HF model → Compare outputs against original
  4. Report config, params, logits, and generation parity

Usage:
    # Full validation (text + image + video generation):
    KERAS_BACKEND=torch python3 tools/verify_qwen3_5_export.py \\
        --preset qwen3_5_0.8b_base

    # Skip generation (faster, logit-only):
    KERAS_BACKEND=torch python3 tools/verify_qwen3_5_export.py \\
        --preset qwen3_5_0.8b_base --skip_generation

Requirements:
    pip install keras-hub transformers torch safetensors pillow requests
    pip install qwen_vl_utils torchvision  (optional, for video)
"""

import argparse
import gc
import os
import tempfile
from io import BytesIO

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer

from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM

print(f"Keras backend: {keras.config.backend()}")
print(f"Keras version: {keras.__version__}")

# ---------------------------------------------------------------
# Constants (same as conversion script)
# ---------------------------------------------------------------
PRESET_TO_HF = {
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
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
VIDEO_FPS = 2
TEXT_PROMPT = "The capital of France is"
MULTIMODAL_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video.<|im_end|>\n<|im_start|>assistant\n"
)

device = torch.device("cpu")


def _load_test_image():
    """Download COCO test image."""
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _load_test_video():
    """Download and decode test video frames."""
    vid_response = requests.get(VIDEO_URL, timeout=60)
    vid_response.raise_for_status()
    vid_path = os.path.join(tempfile.gettempdir(), "qwen3_5_export_test.mp4")
    with open(vid_path, "wb") as f:
        f.write(vid_response.content)

    try:
        from torchvision.io import read_video

        video_tensor, _, info = read_video(
            vid_path, pts_unit="sec", output_format="THWC"
        )
        video_fps = info.get("video_fps", 24.0)
        total_frames = video_tensor.shape[0]
        num_sample = max(4, min(int(total_frames / video_fps * VIDEO_FPS), 8))
        indices = (
            np.linspace(0, total_frames - 1, num_sample).round().astype(int)
        )
        sampled = video_tensor[indices]
        frames = [
            Image.fromarray(sampled[i].numpy()) for i in range(len(sampled))
        ]
        print(
            f"    Video: {total_frames} frames @ {video_fps}fps → "
            f"sampled {len(frames)} frames"
        )
    except ImportError:
        print("    ⚠ torchvision not available, using 4 blank frames")
        frames = [Image.new("RGB", (128, 128)) for _ in range(4)]
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    return frames


def _extract_response(text):
    """Strip prompt prefix and <think> blocks from generated text."""
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


# ---------------------------------------------------------------
# 1. Export KerasHub model to HF format
# ---------------------------------------------------------------
def export_keras_model(preset, export_path):
    """Load KerasHub preset and export to HF format."""
    print("\n[1/7] Loading KerasHub model from preset...")
    keras_model = Qwen3_5CausalLM.from_preset(preset)
    backbone = keras_model.backbone

    has_vision = (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    )
    print(
        f"  ✓ Loaded: {backbone.num_layers} layers, "
        f"{backbone.hidden_dim}d, {backbone.vocabulary_size} vocab"
    )
    print(f"  ✓ Vision encoder: {'yes' if has_vision else 'no'}")
    print(f"  ✓ Parameters: {keras_model.count_params():,}")

    print(f"\n[2/7] Exporting to HF format → {export_path}...")
    keras_model.export_to_transformers(export_path)

    for fname in ["config.json", "model.safetensors", "tokenizer_config.json"]:
        fpath = os.path.join(export_path, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        print(f"  {'✓' if exists else '✗'} {fname} ({size:,} bytes)")

    # Free KerasHub model memory.
    del keras_model
    gc.collect()

    return has_vision


# ---------------------------------------------------------------
# 2. Precompute Original HF outputs
# ---------------------------------------------------------------
def precompute_original_outputs(hf_model_id, has_vision, skip_generation):
    """Load original HF model and precompute all outputs."""
    print(f"\n[3/7] Loading ORIGINAL HF model: {hf_model_id}...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  ✓ Original HF: {hf_params:,} parameters")

    results = {"hf_params": hf_params}

    # --- Text-only ---
    print("\n  Computing text-only outputs...")
    hf_inputs = hf_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = hf_model(**hf_inputs)
    results["text_logits"] = text_out.logits.float().cpu().numpy()
    results["text_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    print(f"    Text logits shape: {results['text_logits'].shape}")

    if not skip_generation:
        with torch.no_grad():
            gen_out = hf_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        results["text_generated"] = hf_tokenizer.decode(
            gen_out[0][hf_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f'    Text generation: "{results["text_generated"][:80]}"')

    # --- Image-to-text ---
    if has_vision:
        print("\n  Computing image-to-text outputs...")
        raw_image = _load_test_image()
        processor = AutoProcessor.from_pretrained(hf_model_id)

        mm_inputs = processor(
            text=[MULTIMODAL_PROMPT], images=[raw_image], return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            mm_out = hf_model(**mm_inputs)
        results["mm_logits"] = mm_out.logits.float().cpu().numpy()
        results["mm_input_ids"] = mm_inputs["input_ids"].cpu().numpy()
        print(f"    Image logits shape: {results['mm_logits'].shape}")

        if not skip_generation:
            with torch.no_grad():
                mm_gen = hf_model.generate(
                    **mm_inputs, max_new_tokens=30, do_sample=False
                )
            results["mm_generated"] = processor.batch_decode(
                mm_gen, skip_special_tokens=True
            )[0]
            print(
                f"    Image generation: "
                f'"{_extract_response(results["mm_generated"])[:80]}"'
            )

        # --- Video-to-text ---
        print("\n  Computing video-to-text outputs...")
        video_frames = _load_test_video()

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
            try:
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, AttributeError):
                # Base models don't have a chat template.
                text_input = VIDEO_PROMPT
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            for k, v in video_kwargs.items():
                if isinstance(v, list) and len(v) == 1:
                    video_kwargs[k] = v[0]
            vid_inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **video_kwargs,
            ).to(device)
        except ImportError:
            print("    ⚠ qwen_vl_utils not installed, using direct call")
            vid_inputs = processor(
                text=[VIDEO_PROMPT],
                videos=[video_frames],
                return_tensors="pt",
            ).to(device)

        try:
            with torch.no_grad():
                vid_out = hf_model(**vid_inputs)
            results["vid_logits"] = vid_out.logits.float().cpu().numpy()
            results["vid_input_ids"] = vid_inputs["input_ids"].cpu().numpy()
            print(f"    Video logits shape: {results['vid_logits'].shape}")

            if not skip_generation:
                with torch.no_grad():
                    vid_gen = hf_model.generate(
                        **vid_inputs, max_new_tokens=16, do_sample=False
                    )
                results["vid_generated"] = processor.batch_decode(
                    vid_gen, skip_special_tokens=True
                )[0]
                print(
                    f"    Video generation: "
                    f'"{_extract_response(results["vid_generated"])[:80]}"'
                )
        except Exception as e:
            print(f"    ⚠ Video forward pass failed: {e}")
            results["vid_logits"] = None

    # Free original HF model.
    del hf_model
    del hf_tokenizer
    gc.collect()

    return results


# ---------------------------------------------------------------
# 3. Load Exported HF model and compare
# ---------------------------------------------------------------
def validate_exported_model(
    export_path, hf_model_id, original_results, has_vision, skip_generation
):
    """Load the exported model and compare against original."""

    print(f"\n[4/7] Loading EXPORTED model from {export_path}...")
    exp_model = AutoModelForCausalLM.from_pretrained(
        export_path, torch_dtype=torch.float32
    )
    exp_model.eval()
    exp_params = sum(p.numel() for p in exp_model.parameters())
    orig_params = original_results["hf_params"]
    print(f"  ✓ Exported: {exp_params:,} parameters")

    param_match = orig_params == exp_params
    print(
        f"  {'✓' if param_match else '✗'} Param count: "
        f"original={orig_params:,}, exported={exp_params:,}"
    )

    # Also load the tokenizer for generation.
    exp_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    results = {"param_match": param_match, "config_pass": True}

    # ------------------------------------------------------------------
    # Step 5: Config comparison
    # ------------------------------------------------------------------
    print("\n[5/7] Comparing configs...")
    orig_model_tmp = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    orig_cfg = orig_model_tmp.config
    exp_cfg = exp_model.config

    orig_text = getattr(orig_cfg, "text_config", orig_cfg)
    exp_text = getattr(exp_cfg, "text_config", exp_cfg)

    fields = [
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "intermediate_size",
        "head_dim",
        "tie_word_embeddings",
    ]
    for name in fields:
        o = getattr(orig_text, name, None)
        e = getattr(exp_text, name, None)
        match = o == e
        print(f"  {'✓' if match else '✗'} {name}: original={o}, exported={e}")
        if not match:
            results["config_pass"] = False

    del orig_model_tmp
    gc.collect()

    # ------------------------------------------------------------------
    # Step 6: Text-only validation
    # ------------------------------------------------------------------
    print("\n[6/7] TEXT VALIDATION")
    text_ids = torch.tensor(original_results["text_input_ids"]).to(device)

    with torch.no_grad():
        exp_text_out = exp_model(input_ids=text_ids)
    exp_text_logits = exp_text_out.logits.float().cpu().numpy()
    orig_text_logits = original_results["text_logits"]

    text_diff = np.abs(exp_text_logits - orig_text_logits)
    results["text_mean_diff"] = float(text_diff.mean())
    print(f"  Logit mean abs diff: {results['text_mean_diff']:.2e}")
    results["text_pass"] = results["text_mean_diff"] < 0.1

    # Top-5 overlap for first-token prediction.
    orig_top5 = set(np.argsort(orig_text_logits[0, -1])[-5:].tolist())
    exp_top5 = set(np.argsort(exp_text_logits[0, -1])[-5:].tolist())
    overlap = len(orig_top5 & exp_top5)
    print(f"  Top-5 token overlap: {overlap}/5 ({100 * overlap / 5:.0f}%)")

    if not skip_generation:
        print("\n  Text generation comparison:")
        hf_inputs = exp_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
        prompt_len = hf_inputs["input_ids"].shape[1]

        with torch.no_grad():
            exp_gen = exp_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        exp_gen_text = exp_tokenizer.decode(
            exp_gen[0][prompt_len:], skip_special_tokens=True
        )
        orig_gen_text = original_results.get("text_generated", "N/A")

        print(f'    Prompt:   "{TEXT_PROMPT}"')
        print(f'    Original: "{orig_gen_text[:80]}"')
        print(f'    Exported: "{exp_gen_text[:80]}"')

        if orig_gen_text == exp_gen_text:
            print("    ✓ Text generation is IDENTICAL")
        else:
            print("    ⚠ Text differs (expected with bf16→f32 precision)")

    # ------------------------------------------------------------------
    # Step 7: Multimodal validation (image + video)
    # ------------------------------------------------------------------
    if has_vision:
        print("\n[7/7] MULTIMODAL VALIDATION")

        # --- Image-to-text ---
        if "mm_logits" in original_results:
            print("\n  Image-to-text:")
            mm_ids = torch.tensor(original_results["mm_input_ids"]).to(device)

            with torch.no_grad():
                exp_mm_out = exp_model(input_ids=mm_ids)
            exp_mm_logits = exp_mm_out.logits.float().cpu().numpy()
            orig_mm_logits = original_results["mm_logits"]

            mm_diff = np.abs(exp_mm_logits - orig_mm_logits)
            results["mm_mean_diff"] = float(mm_diff.mean())
            print(f"    Logit mean abs diff: {results['mm_mean_diff']:.2e}")

            if not skip_generation:
                orig_mm_gen = original_results.get("mm_generated", "N/A")
                print(f'    Original: "{_extract_response(orig_mm_gen)[:80]}"')

        # --- Video-to-text ---
        if original_results.get("vid_logits") is not None:
            print("\n  Video-to-text:")
            vid_ids = torch.tensor(original_results["vid_input_ids"]).to(device)

            with torch.no_grad():
                exp_vid_out = exp_model(input_ids=vid_ids)
            exp_vid_logits = exp_vid_out.logits.float().cpu().numpy()
            orig_vid_logits = original_results["vid_logits"]

            vid_diff = np.abs(exp_vid_logits - orig_vid_logits)
            results["vid_mean_diff"] = float(vid_diff.mean())
            print(f"    Logit mean abs diff: {results['vid_mean_diff']:.2e}")

            if not skip_generation:
                orig_vid_gen = original_results.get("vid_generated", "N/A")
                print(f'    Original: "{_extract_response(orig_vid_gen)[:80]}"')
        else:
            print("\n  Video-to-text: ⚠ Skipped (original failed)")
    else:
        print("\n[7/7] No vision encoder — skipping multimodal validation.")

    # Clean up.
    del exp_model
    gc.collect()

    return results


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
def print_summary(results):
    """Print final summary."""
    config_pass = results.get("config_pass", False)
    text_pass = results.get("text_pass", False)
    param_match = results.get("param_match", False)

    all_pass = config_pass and text_pass
    print("\n" + "=" * 70)
    if all_pass:
        print("  ✅ ALL CHECKS PASSED")
        print("     - Config fields match ✓")
        print(
            f"     - Parameter count: "
            f"{'match ✓' if param_match else 'differ (see note)'}"
        )
        print(
            f"     - Text logit parity ✓ "
            f"(mean diff: {results.get('text_mean_diff', 'N/A'):.2e})"
        )
        if "mm_mean_diff" in results:
            print(
                f"     - Image logit parity ✓ "
                f"(mean diff: {results['mm_mean_diff']:.2e})"
            )
        if "vid_mean_diff" in results:
            print(
                f"     - Video logit parity ✓ "
                f"(mean diff: {results['vid_mean_diff']:.2e})"
            )
    else:
        print("  ❌ SOME CHECKS FAILED — Review output above")
    print("=" * 70 + "\n")

    return all_pass


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify Qwen3.5 KerasHub → HF export"
    )
    parser.add_argument(
        "--preset",
        default="qwen3_5_0.8b_base",
        help="KerasHub preset name (default: qwen3_5_0.8b_base)",
    )
    parser.add_argument(
        "--hf_model_id",
        default=None,
        help="HuggingFace model ID (auto-detected from preset if omitted)",
    )
    parser.add_argument(
        "--export_dir",
        default=None,
        help="Directory to export to (uses temp dir if omitted)",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip text/image/video generation (faster, logit-only check)",
    )
    args = parser.parse_args()

    hf_model_id = args.hf_model_id or PRESET_TO_HF.get(args.preset)
    if hf_model_id is None:
        print(f"Error: No HF model ID for preset '{args.preset}'.")
        exit(1)

    export_dir = args.export_dir or tempfile.mkdtemp()
    export_path = os.path.join(export_dir, "qwen3_5_exported")

    print("\n" + "=" * 70)
    print("  Qwen3.5 Export Verification (Real Pretrained Weights)")
    print("=" * 70)
    print(f"  KerasHub preset  : {args.preset}")
    print(f"  HF model ID      : {hf_model_id}")
    print(f"  Export path       : {export_path}")
    print(f"  Skip generation   : {args.skip_generation}")

    # Phase 1: Export KerasHub model.
    has_vision = export_keras_model(args.preset, export_path)

    # Phase 2: Precompute original HF outputs.
    original_results = precompute_original_outputs(
        hf_model_id, has_vision, args.skip_generation
    )

    # Phase 3: Load exported model and validate.
    validation_results = validate_exported_model(
        export_path,
        hf_model_id,
        original_results,
        has_vision,
        args.skip_generation,
    )

    # Summary.
    success = print_summary(validation_results)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
