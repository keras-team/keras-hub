"""Convert Qwen2.5-VL checkpoints from Hugging Face to KerasHub presets.

Uses the maintained `hf://` loading path (utils/transformers/convert_qwen2_5_vl)
for the actual conversion, then validates the result against the HF model:
parameter count, tokenization, text, image and video logits, and generation.

    python tools/checkpoint_conversion/convert_qwen2_5_vl_checkpoints.py \
        --preset qwen2_5_vl_7b_instruct
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import gc  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from keras import ops  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402
from tools.checkpoint_conversion.checkpoint_conversion_utils import (  # noqa: E402
    extract_response,
)
from tools.checkpoint_conversion.checkpoint_conversion_utils import (  # noqa: E402
    save_preset,
)
from tools.checkpoint_conversion.checkpoint_conversion_utils import (  # noqa: E402
    test_parameter_count,
)
from tools.checkpoint_conversion.checkpoint_conversion_utils import (  # noqa: E402
    validate_reloaded_preset,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

PRESET_MAP = {
    "qwen2_5_vl_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_32b_instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
}

TEXT_PROMPT = "What is the capital of France?"
MULTIMODAL_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|>"
    "<|vision_end|>Describe this image."
    "<|im_end|>\n<|im_start|>assistant\n"
)
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|>"
    "<|vision_end|>Describe this video."
    "<|im_end|>\n<|im_start|>assistant\n"
)
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/"
    "documentation-images/resolve/main/transformers/tasks/car.jpg"
)
VIDEO_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/"
    "fixtures_videos/resolve/main/tennis.mp4"
)
NUM_VIDEO_FRAMES = 8

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


def _load_test_image():
    path = keras.utils.get_file("qwen2_5_vl_test_car.jpg", IMAGE_URL)
    return Image.open(path).convert("RGB")


def _load_test_video():
    """Decode the fixture clip to (T, H, W, C) uint8 plus fps metadata."""
    import av

    path = keras.utils.get_file("qwen2_5_vl_test_video.mp4", VIDEO_URL)
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = [
        frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
    ]
    container.close()
    video = np.stack(frames)
    # Deterministic uniform subsample so both processors see identical
    # frames.
    idx = np.linspace(0, len(video) - 1, NUM_VIDEO_FRAMES).astype(np.int64)
    return video[idx], fps


def _hf_mm_inputs(processor_output):
    """Tensorize an AutoProcessor output for the HF forward call."""
    return {
        key: value.to(device)
        for key, value in processor_output.items()
        if torch.is_tensor(value)
    }


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, hf_tokenizer, hf_processor, hf_preset):
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

    # --- Image outputs (end-to-end AutoProcessor) ---
    raw_image = _load_test_image()
    image_inputs = hf_processor(
        text=[MULTIMODAL_PROMPT], images=[raw_image], return_tensors="pt"
    )
    hf_inputs = _hf_mm_inputs(image_inputs)
    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)
    results["mm_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["mm_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    results["mm_pixel_values"] = hf_inputs["pixel_values"].cpu().float().numpy()
    results["mm_grid_thw"] = hf_inputs["image_grid_thw"].cpu().numpy()
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

    # --- Video outputs (end-to-end AutoProcessor + second_per_grid_ts) ---
    frames, fps = _load_test_video()
    video_inputs = hf_processor(
        text=[VIDEO_PROMPT],
        videos=[frames],
        video_metadata=[
            {
                "fps": fps,
                "duration": len(frames) / fps,
                "total_num_frames": len(frames),
            }
        ],
        return_tensors="pt",
    )
    hf_inputs = _hf_mm_inputs(video_inputs)
    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)
    results["video_logits"] = hf_out.logits.detach().cpu().float().numpy()
    results["video_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    results["video_pixel_values"] = (
        hf_inputs["pixel_values_videos"].cpu().float().numpy()
    )
    results["video_grid_thw"] = hf_inputs["video_grid_thw"].cpu().numpy()
    if "second_per_grid_ts" in video_inputs:
        results["second_per_grid_ts"] = (
            video_inputs["second_per_grid_ts"].cpu().numpy().tolist()
        )
    else:
        # HF's documented fallback when fps metadata is unavailable.
        results["second_per_grid_ts"] = [1.0]
    results["video_frames"] = frames

    return results


# ---------------------------------------------------------------
# 2. Text-only validation
# ---------------------------------------------------------------
def validate_text_output(keras_model, hf_results):
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION")
    print("=" * 50)

    hf_ids = hf_results["text_token_ids"]
    keras_preprocessed = keras_model.preprocessor.generate_preprocess(
        [TEXT_PROMPT], sequence_length=hf_ids.shape[1]
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
    keras_valid = keras_ids[keras_mask.astype(bool)]
    np.testing.assert_array_equal(keras_valid, hf_ids[0])
    print("  Token IDs match.")

    with torch.no_grad():
        keras_logits = ops.convert_to_numpy(
            keras_model(keras_preprocessed)
        ).astype(np.float32)

    hf_logits = hf_results["text_logits"]
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
    print("  Logits match within atol=1e-3.")

    if not FLAGS.skip_generation:
        with torch.no_grad():
            keras_output = keras_model.generate(
                TEXT_PROMPT, max_length=int(hf_ids.shape[1]) + 32
            )
        keras_text = extract_response(keras_output)
        hf_text = extract_response(hf_results["text_generated"])
        print(f"  KerasHub: {keras_text}")
        print(f"  HF:       {hf_text}")
        if keras_text != hf_text:
            # Greedy diverges at near-tie logits.
            print("  WARNING: greedy generation differs.")
        else:
            print("  Greedy generation matches.")


# ---------------------------------------------------------------
# 3. Multimodal validation
# ---------------------------------------------------------------
def _compare_media_parity(
    keras_model,
    keras_preprocessed,
    hf_pixel_values,
    hf_input_ids,
    hf_grid_thw,
    hf_logits,
    label,
    logit_atol,
):
    """Assert grid/token/logit parity for one media-modality forward."""
    ve = keras_model.backbone.vision_encoder
    C, T = ve.in_channels, ve.temporal_patch_size
    pH = pW = ve.patch_size
    # HF pixel_values is (N, C*T*pH*pW) flattened; KerasHub uses
    # (N, T, pH, pW, C).
    pixel_values_np = hf_pixel_values.reshape(-1, C, T, pH, pW)
    pixel_values_np = np.transpose(pixel_values_np, (0, 2, 3, 4, 1))

    # Single-media preprocess emits the combined grid under
    # "image_grid_thw" for both modalities.
    keras_grid = ops.convert_to_numpy(keras_preprocessed["image_grid_thw"])
    np.testing.assert_array_equal(keras_grid, hf_grid_thw)
    print(f"  {label} grid_thw matches HF.")

    keras_pixels = ops.convert_to_numpy(keras_preprocessed["pixel_values"])
    np.testing.assert_allclose(keras_pixels, pixel_values_np, atol=1e-5)
    print(f"  {label} pixels match HF processor (atol=1e-5).")

    with torch.no_grad():
        keras_logits = ops.convert_to_numpy(
            keras_model(keras_preprocessed)
        ).astype(np.float32)

    # Sequence lengths must match exactly — no common-prefix slicing.
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
    keras_valid = int(keras_mask.sum())
    assert keras_valid == hf_input_ids.shape[1], (
        f"{label} sequence length mismatch: "
        f"keras={keras_valid}, hf={hf_input_ids.shape[1]}"
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    np.testing.assert_array_equal(keras_ids[:, :keras_valid], hf_input_ids)
    print(f"  {label} token IDs match.")

    keras_real = keras_logits[:, :keras_valid]
    hf_real = hf_logits[:, :keras_valid]
    abs_diff = np.abs(keras_real - hf_real)
    print(f"  {label} logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  {label} logit max absolute diff:  {abs_diff.max():.6f}")
    # HF sdpa vs eager vision features differ ~6e-3 on identical input;
    # Qwen2.5-VL-3B logit max diffs: image 0.03, video 0.24.
    np.testing.assert_allclose(keras_real, hf_real, atol=logit_atol)
    print(f"  {label} logits match within atol={logit_atol}.")


def validate_multimodal_output(keras_model, hf_results):
    if keras_model.backbone.vision_encoder is None:
        print("\n-> Skipping multimodal validation (no vision encoder).")
        return

    print("\n" + "=" * 50)
    print("MULTIMODAL VALIDATION")
    print("=" * 50)

    image_preprocessed = keras_model.preprocessor.generate_preprocess(
        {
            "prompts": [MULTIMODAL_PROMPT],
            "images": [np.asarray(hf_results["raw_image"])],
        },
        sequence_length=int(hf_results["mm_input_ids"].shape[1]),
    )
    _compare_media_parity(
        keras_model,
        image_preprocessed,
        hf_results["mm_pixel_values"],
        hf_results["mm_input_ids"],
        hf_results["mm_grid_thw"],
        hf_results["mm_logits"],
        "image",
        0.1,
    )

    video_preprocessed = keras_model.preprocessor.generate_preprocess(
        {
            "prompts": [VIDEO_PROMPT],
            "videos": [hf_results["video_frames"]],
            "second_per_grid_ts": hf_results["second_per_grid_ts"],
        },
        sequence_length=int(hf_results["video_input_ids"].shape[1]),
    )
    _compare_media_parity(
        keras_model,
        video_preprocessed,
        hf_results["video_pixel_values"],
        hf_results["video_input_ids"],
        hf_results["video_grid_thw"],
        hf_results["video_logits"],
        "video",
        0.5,
    )

    if not FLAGS.skip_generation:
        with torch.no_grad():
            keras_output = keras_model.generate(
                {
                    "prompts": [MULTIMODAL_PROMPT],
                    "images": [np.array(hf_results["raw_image"])],
                },
                max_length=int(hf_results["mm_input_ids"].shape[1]) + 32,
            )
        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        keras_text = extract_response(keras_text)
        hf_text = extract_response(hf_results["mm_generated"])
        print(f"  KerasHub output: {keras_text}")
        print(f"  HF output:       {hf_text}")
        if keras_text != hf_text:
            # Greedy diverges at near-tie logits.
            print("  WARNING: greedy generation differs.")
        else:
            print("  Greedy generation matches.")


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
    hf_processor = AutoProcessor.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(
        hf_model, hf_tokenizer, hf_processor, hf_preset
    )

    # --- Phase 2: Free HF model, load KerasHub model ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.Qwen2_5VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    keras_model.compile(sampler="greedy")

    # --- Phase 3: Validate ---
    test_parameter_count(keras_model.backbone, hf_params)
    validate_text_output(keras_model, hf_results)
    validate_multimodal_output(keras_model, hf_results)

    # --- Phase 4: Save and validate the artifact ---
    del keras_model
    del hf_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n-> Reloading model in bfloat16 for saving...")
    save_model = keras_hub.models.Qwen2_5VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    preset_path = save_preset(save_model, preset)
    del save_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    validate_reloaded_preset(keras_hub.models.Qwen2_5VLCausalLM, preset_path)
    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
