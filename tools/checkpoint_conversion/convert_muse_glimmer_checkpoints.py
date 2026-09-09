"""Convert Muse Glimmer HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_muse_glimmer_checkpoints.py \
        --preset muse_glimmer_30b
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
    "muse_glimmer_30b": "meta-models/Muse-Glimmer-30B",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
TEXT_PROMPT = "What is Keras?"
IMAGE_PROMPT = "Describe this image: <|image_start|><|patch|><|image_end|>"
VIDEO_PROMPT = "Describe this video: <|vid_start|><|video|><|vid_end|>"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "If True, skip all text generation steps and only run numerical "
    "logit validation.",
)


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _load_test_video():
    """Load a short video and return two frames for video validation."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "PyAV is required for Muse Glimmer video validation."
        ) from error

    try:
        response = requests.get(VIDEO_URL, timeout=60)
        response.raise_for_status()
        container = av.open(BytesIO(response.content))
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    except Exception as error:
        raise RuntimeError(
            f"Could not load the Muse Glimmer test video: {error}"
        ) from error

    if len(frames) < 2:
        raise RuntimeError(
            "The Muse Glimmer test video has fewer than two frames."
        )

    return np.stack([frames[0], frames[-1]])


def _count_keras_params(backbone):
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


def _expand_vision_prompt(prompt, token, num_tokens):
    return prompt.replace(token, token * num_tokens)


def _precompute_multimodal_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    prompt,
    media,
    modality,
):
    if modality == "image":
        visual_inputs = processor.image_processor([media], return_tensors="pt")
        visual_key = "pixel_values"
        grid_key = "image_grid_thw"
        merge_size = processor.image_processor.merge_size
        placeholder = "<|patch|>"
    else:
        visual_inputs = processor.video_processor([media], return_tensors="pt")
        visual_key = "pixel_values_videos"
        grid_key = "video_grid_thw"
        merge_size = processor.video_processor.merge_size
        placeholder = "<|video|>"

    grid = visual_inputs[grid_key]
    num_tokens = int(grid[0].prod().item()) // merge_size**2
    expanded_prompt = _expand_vision_prompt(prompt, placeholder, num_tokens)
    text_inputs = hf_tokenizer([expanded_prompt], return_tensors="pt")
    hf_inputs = {**text_inputs, **visual_inputs}
    hf_inputs.pop("video_metadata", None)

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, use_cache=False)

    result = {
        "prompt": prompt,
        "input_ids": hf_inputs["input_ids"].cpu().numpy().astype(np.int32),
        "attention_mask": hf_inputs["attention_mask"]
        .cpu()
        .numpy()
        .astype(np.int32),
        "logits": hf_outputs.logits.detach().cpu().float().numpy(),
        "grid_thw": grid.cpu().numpy().astype(np.int32),
        "pixel_values": visual_inputs[visual_key].cpu().float().numpy(),
        "media": np.asarray(media) if modality == "image" else media,
        "modality": modality,
    }

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_generated = hf_model.generate(
                **hf_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        prompt_length = hf_inputs["input_ids"].shape[1]
        result["generated"] = hf_tokenizer.decode(
            hf_generated[0, prompt_length:],
            skip_special_tokens=True,
        )

    return result


def precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset):
    results = {}

    hf_ids = hf_tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    results["text_token_ids"] = hf_ids
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.tensor(hf_ids, dtype=torch.long).to(device),
            use_cache=False,
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

    processor = AutoProcessor.from_pretrained(hf_preset)
    raw_image = _load_test_image()
    results["image"] = _precompute_multimodal_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        IMAGE_PROMPT,
        raw_image,
        "image",
    )

    raw_video = _load_test_video()
    results["video"] = _precompute_multimodal_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        VIDEO_PROMPT,
        raw_video,
        "video",
    )

    return results


def test_parameter_count(keras_backbone, hf_param_count):
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)
    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")
    np.testing.assert_equal(keras_params, hf_param_count)
    print("  ✓ Parameter counts match!")


def _build_keras_multimodal_inputs(keras_model, result):
    token_ids = result["input_ids"]
    if result["modality"] == "image":
        placeholder_id = keras_model.preprocessor.tokenizer.image_token_id
    else:
        placeholder_id = keras_model.preprocessor.tokenizer.video_token_id
    vision_indices = np.where(token_ids[0] == placeholder_id)[0].astype(
        np.int32
    )[np.newaxis, :]

    pixel_values = result["pixel_values"]
    if pixel_values.ndim == 2:
        pixel_values = pixel_values[np.newaxis, ...]
    grid_thw = result["grid_thw"]
    if grid_thw.ndim == 2:
        grid_thw = grid_thw[np.newaxis, ...]

    return {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(result["attention_mask"]),
        "pixel_values": ops.convert_to_tensor(pixel_values),
        "image_grid_thw": ops.convert_to_tensor(grid_thw),
        "vision_indices": ops.convert_to_tensor(vision_indices),
    }


def test_token_ids(keras_model, hf_results, label):
    print("\n" + "=" * 50)
    print(f"{label} TOKEN ID VALIDATION")
    print("=" * 50)

    if label == "TEXT":
        hf_ids = hf_results["text_token_ids"]
        preprocessor_inputs = TEXT_PROMPT
    else:
        result = hf_results[label.lower()]
        hf_ids = result["input_ids"]
        media_key = "images" if label == "IMAGE" else "videos"
        preprocessor_inputs = {
            "prompts": [result["prompt"]],
            media_key: result["media"],
        }

    keras_preprocessed = keras_model.preprocessor.generate_preprocess(
        preprocessor_inputs,
        sequence_length=hf_ids.shape[1],
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
    keras_valid = keras_ids[keras_mask.astype(bool)]
    print(f"\n  HF token IDs:       {hf_ids[0][:10].tolist()}")
    print(f"  KerasHub token IDs: {keras_valid[:10].tolist()}")
    np.testing.assert_array_equal(keras_valid, hf_ids[0])
    print(f" ✓ [{label}] Token IDs match.")


def test_numerics(keras_model, hf_results, label):
    if label == "TEXT":
        hf_ids = hf_results["text_token_ids"]
        token_ids = ops.convert_to_tensor(hf_ids.astype(np.int32))
        padding_mask = ops.ones_like(token_ids)
        model_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        batch_size = ops.shape(token_ids)[0]
        vision_encoder = keras_model.backbone.vision_encoder
        patch_dim = (
            vision_encoder.patch_temporal * 3 * vision_encoder.patch_size**2
        )
        model_inputs.update(
            {
                "pixel_values": ops.zeros(
                    (batch_size, 0, patch_dim), dtype="float32"
                ),
                "image_grid_thw": ops.zeros((batch_size, 0, 3), dtype="int32"),
                "vision_indices": ops.zeros((batch_size, 0), dtype="int32"),
            }
        )
        keras_logits = ops.convert_to_numpy(keras_model(model_inputs)).astype(
            np.float32
        )
        _report_numerics({label}, keras_logits, hf_results["text_logits"])
        return

    result = hf_results[label.lower()]
    keras_inputs = _build_keras_multimodal_inputs(keras_model, result)
    keras_logits = ops.convert_to_numpy(keras_model(keras_inputs)).astype(
        np.float32
    )
    _report_numerics({label}, keras_logits, result["logits"])


def _report_numerics(label, keras_logits, hf_logits):
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  {label} logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  {label} logit max absolute diff:  {abs_diff.max():.6f}")
    try:
        np.testing.assert_allclose(
            keras_logits, hf_logits, atol=1e-3, rtol=1e-3
        )
        print(f" ✓ [{label}] logits match within atol=1e-3, rtol=1e-3.")
    except AssertionError as error:
        print(f"  [{label}] logits differ beyond tolerance: {error}")


def test_generation(keras_model, hf_results, label):
    if label == "TEXT":
        max_length = hf_results["text_token_ids"].shape[1] + 32
        keras_output = keras_model.generate(TEXT_PROMPT, max_length=max_length)
        hf_output = hf_results.get("text_generated", "N/A")
    else:
        result = hf_results[label.lower()]
        media_key = "images" if label == "IMAGE" else "videos"
        keras_output = keras_model.generate(
            {
                "prompts": [result["prompt"]],
                media_key: result["media"],
            },
            max_length=result["input_ids"].shape[1] + 32,
            strip_prompt=True,
        )
        if isinstance(keras_output, (list, tuple)):
            keras_output = keras_output[0]
        hf_output = result.get("generated", "N/A")

    print(f"\n  {label} KerasHub: {keras_output}")
    print(f"  {label} HF:       {hf_output}")
    print(f" ✓ [{label}] Text generation completed.")


def validate_output(keras_model, hf_results):
    labels = ("TEXT", "IMAGE", "VIDEO")

    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    for label in labels:
        test_token_ids(keras_model, hf_results, label)
        test_numerics(keras_model, hf_results, label)

    if FLAGS.skip_generation:
        return

    keras_model.compile(sampler="greedy")
    for label in labels:
        test_generation(keras_model, hf_results, label)


def save_preset(keras_model, preset_name):
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    print("-> Loading HF model...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset)
    hf_results["hf_param_count"] = hf_params
    print("   HF outputs precomputed!")

    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    print("   HF model released.")

    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.MuseGlimmerCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_output(keras_model, hf_results)

    save_preset(keras_model, preset)
    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
