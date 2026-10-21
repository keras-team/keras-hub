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
from transformers.video_utils import VideoMetadata

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
# End-to-end top-1 agreement with HF logits across the prompt.
MIN_TOP1_AGREEMENT = 0.95
TEXT_PROMPT = "What is Keras?"
MULTIMODAL_TEXT = "Describe this image in detail."
VIDEO_TEXT = "Describe what is happening in this video."
# KerasHub prompt with <image> placeholder and chat formatting.
KERASHUB_MULTIMODAL_PROMPT = (
    "<|im_start|>User:<image>"
    + MULTIMODAL_TEXT
    + "<end_of_utterance>\nAssistant:"
)
# HF's chat template puts a space before a video, but not before an image.
KERASHUB_VIDEO_PROMPT = (
    "<|im_start|>User: <video>" + VIDEO_TEXT + "<end_of_utterance>\nAssistant:"
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
flags.DEFINE_bool(
    "skip_video_validation",
    False,
    "If True, skip the video checks. Without it, a missing video decoder "
    "is an error rather than a silent fallback.",
)


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


def _decode_video(path):
    """Decode an MP4 into `(THWC uint8 frames, fps)`, failing loudly.

    Uses TorchCodec, which torchvision now points to for video decoding
    (`torchvision.io.read_video` was removed). TorchCodec needs FFmpeg
    installed on the system.
    """
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise RuntimeError(
            "Video validation needs `torchcodec`, which also needs FFmpeg. "
            "Install both, or pass `--skip_video_validation` to skip the "
            "video checks explicitly."
        ) from e
    decoder = VideoDecoder(path, dimension_order="NHWC")
    return decoder[:], decoder.metadata.average_fps


def _hf_pixels_to_channels_last(pixel_values):
    """HF `(batch, images, C, H, W)` to KerasHub `(batch*images, H, W, C)`."""
    if pixel_values.ndim == 5:
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(b * n, c, h, w)
    return np.transpose(pixel_values, (0, 2, 3, 1))


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

    if FLAGS.skip_video_validation:
        print("\n   ⚠ Skipping video outputs (--skip_video_validation).")
        return results

    # --- Video outputs ---
    # Download and decode real MP4 video.
    print("\n   Downloading test video...")
    vid_response = requests.get(VIDEO_URL, timeout=60)
    vid_response.raise_for_status()
    vid_path = os.path.join(tempfile.gettempdir(), "smolvlm2_test_video.mp4")
    with open(vid_path, "wb") as f:
        f.write(vid_response.content)

    try:
        video_tensor, video_fps = _decode_video(vid_path)
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)
    total_frames = video_tensor.shape[0]

    # Sample frames at VIDEO_FPS (SmolVLM2 default = 1 fps).
    # Cap at 8 to avoid OOM on limited-memory devices.
    num_sample = max(4, min(int(total_frames / video_fps * VIDEO_FPS), 8))
    indices = np.linspace(0, total_frames - 1, num_sample).round().astype(int)
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
    # Both processors get the same frames and the same metadata, so the
    # timestamps and frame count in the prompts can be compared token by
    # token.
    video_metadata = {
        "fps": float(video_fps),
        "frames_indices": indices.tolist(),
    }

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
        text=video_prompt,
        videos=[[video_frames]],
        video_metadata=[
            VideoMetadata(total_num_frames=len(video_frames), **video_metadata)
        ],
        return_tensors="pt",
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
    results["video_metadata"] = video_metadata

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
    np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
    print("  ✓ Logits match within atol=1e-3.")

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
def _validate_multimodal_parity(keras_model, hf_results, prefix, inputs):
    """Compare KerasHub with HF on token ids, pixels, and logits.

    Token ids come from the KerasHub preprocessor and must match exactly.
    Pixels are only reported: both sides use Lanczos, but KerasHub's
    Lanczos is not bit-identical to PIL's. Logits are checked twice
    through a plain `backbone()` call:

    1. HF `pixel_values` into the KerasHub backbone, asserted at
       `atol=1e-3`. This validates the converted weights.
    2. KerasHub's own pixels, end to end. Top-1 agreement with HF across
       the prompt must be at least `MIN_TOP1_AGREEMENT`.
    """
    hf_ids = hf_results[f"{prefix}_input_ids"]
    preprocessed = keras_model.preprocessor.generate_preprocess(
        inputs, sequence_length=hf_ids.shape[1]
    )

    # --- Token ID parity ---
    keras_ids = ops.convert_to_numpy(preprocessed["token_ids"])
    print(f"\n  HF token count:       {hf_ids.shape[1]}")
    print(f"  KerasHub token count: {keras_ids.shape[1]}")
    np.testing.assert_array_equal(keras_ids, hf_ids)
    print("  ✓ Token IDs match.")

    # --- Pixel comparison (informational) ---
    hf_pixels = _hf_pixels_to_channels_last(
        hf_results[f"{prefix}_pixel_values"]
    )
    keras_pixels = ops.convert_to_numpy(preprocessed["pixel_values"])
    keras_pixels = keras_pixels.astype(np.float32)
    print(f"  HF pixel_values shape:       {hf_pixels.shape}")
    print(f"  KerasHub pixel_values shape: {keras_pixels.shape}")
    # The number of crops must agree, or the token layout would differ too.
    np.testing.assert_equal(
        keras_pixels.shape, hf_pixels.shape, err_msg="Crop count differs."
    )
    pixel_diff = np.abs(keras_pixels - hf_pixels)
    print(
        f"  Pixel mean absolute diff: {pixel_diff.mean():.6f}, "
        f"max: {pixel_diff.max():.6f} (not asserted: Keras Lanczos vs "
        "PIL LANCZOS)"
    )

    backbone = keras_model.backbone
    hf_logits = hf_results[f"{prefix}_logits"]
    padding_mask = ops.convert_to_numpy(preprocessed["padding_mask"])
    last = int(np.nonzero(padding_mask[0])[0][-1])

    def backbone_logits(pixel_values):
        backbone_inputs = {
            key: ops.convert_to_tensor(value)
            for key, value in preprocessed.items()
        }
        backbone_inputs["pixel_values"] = ops.convert_to_tensor(pixel_values)
        backbone_inputs["padding_mask"] = ops.cast(
            backbone_inputs["padding_mask"], "int32"
        )
        hidden = backbone(backbone_inputs)
        logits = backbone.token_embedding(hidden, reverse=True)
        return ops.convert_to_numpy(logits).astype(np.float32)

    # --- 1. Weight parity: HF pixels through the KerasHub backbone ---
    port_diff = np.abs(backbone_logits(hf_pixels) - hf_logits)
    print(f"\n  [HF pixels] logit mean absolute diff: {port_diff.mean():.6f}")
    print(f"  [HF pixels] logit max absolute diff:  {port_diff.max():.6f}")
    np.testing.assert_allclose(port_diff, 0.0, atol=1e-3)
    print("  ✓ Logits on HF pixels match within atol=1e-3.")

    # --- 2. End to end: KerasHub pixels ---
    e2e_logits = backbone_logits(keras_pixels)
    e2e_diff = np.abs(e2e_logits - hf_logits)
    agree = np.mean(
        np.argmax(e2e_logits[0], -1)[padding_mask[0].astype(bool)]
        == np.argmax(hf_logits[0], -1)[padding_mask[0].astype(bool)]
    )
    print(f"\n  [KH pixels] logit mean absolute diff: {e2e_diff.mean():.6f}")
    print(f"  [KH pixels] logit max absolute diff:  {e2e_diff.max():.6f}")
    print(f"  [KH pixels] top-1 agreement over all positions: {agree:.4f}")
    keras_next = int(np.argmax(e2e_logits[0, last]))
    hf_next = int(np.argmax(hf_logits[0, last]))
    print(f"  [KH pixels] next token: KerasHub {keras_next}, HF {hf_next}")
    if agree < MIN_TOP1_AGREEMENT:
        raise AssertionError(
            f"End-to-end top-1 agreement {agree:.4f} is below "
            f"{MIN_TOP1_AGREEMENT}."
        )
    print(
        f"  ✓ End-to-end top-1 agreement {agree:.4f} >= {MIN_TOP1_AGREEMENT}."
    )


def validate_multimodal_output(keras_model, hf_results):
    """Validate multimodal token IDs, pixels, logits and generation."""
    print("\n" + "=" * 50)
    print("MULTIMODAL VALIDATION")
    print("=" * 50)

    raw_image = np.array(hf_results["raw_image"])
    _validate_multimodal_parity(
        keras_model,
        hf_results,
        prefix="mm",
        inputs={"prompts": KERASHUB_MULTIMODAL_PROMPT, "images": raw_image},
    )

    if not FLAGS.skip_generation:
        # --- End-to-end generation ---
        print(f"\n  HF output: {hf_results.get('mm_generated', 'N/A')}")

        keras_output = keras_model.generate(
            {
                "prompts": [KERASHUB_MULTIMODAL_PROMPT],
                "images": [raw_image],
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
    """Validate video token IDs, pixels, logits and generation."""
    print("\n" + "=" * 50)
    print("VIDEO VALIDATION")
    print("=" * 50)

    video_np = np.stack(
        [np.array(f) for f in hf_results["video_frames"]], axis=0
    )  # (num_frames, H, W, 3)
    video_metadata = hf_results["video_metadata"]
    _validate_multimodal_parity(
        keras_model,
        hf_results,
        prefix="video",
        inputs={
            "prompts": KERASHUB_VIDEO_PROMPT,
            "videos": video_np,
            "video_metadata": video_metadata,
        },
    )

    if not FLAGS.skip_generation:
        # --- End-to-end video generation ---
        print(
            f"\n  HF video output: {hf_results.get('video_generated', 'N/A')}"
        )

        keras_output = keras_model.generate(
            {
                "prompts": [KERASHUB_VIDEO_PROMPT],
                "videos": [video_np],
                "video_metadata": [video_metadata],
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
    if FLAGS.skip_video_validation:
        print("\n⚠ Video validation skipped (--skip_video_validation).")
    else:
        validate_video_output(keras_model, hf_results)

    # --- Phase 5: Save preset ---
    save_preset(keras_model, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
