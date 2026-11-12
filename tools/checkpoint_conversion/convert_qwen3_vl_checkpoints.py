"""Convert Qwen3-VL HuggingFace checkpoints to KerasHub presets.

The HF model runs first on text, an image and a video, and its outputs are
kept. It is then freed, and the KerasHub model is loaded once with
`from_preset("hf://...")`, the path users take. The script compares the
parameter count, the token ids, the logits and greedy generation, then saves
the preset in `--dtype` (bfloat16 by default).

Usage:
    python tools/checkpoint_conversion/convert_qwen3_vl_checkpoints.py \
        --preset qwen3_vl_2b_instruct
"""

import gc
import os
import random
import tempfile
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.set_default_device(torch.device("cpu"))
# True float32 matmuls, so a GPU run stays comparable with the HF reference.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PRESET_MAP = {
    "qwen3_vl_2b_instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_2b_thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "qwen3_vl_4b_instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_4b_thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3_vl_32b_thinking": "Qwen/Qwen3-VL-32B-Thinking",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/"
    "resolve/main/tennis.mp4"
)
VIDEO_FPS = 2
MAX_VIDEO_FRAMES = 8
MAX_NEW_TOKENS = 32
TEXT_PROMPT = "What is Keras?"
IMAGE_QUESTION = "Describe this image."
VIDEO_QUESTION = "Describe this video."
# Logits must match HF within this absolute tolerance at every position.
LOGITS_ATOL = 1e-3

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string("dtype", "bfloat16", "Dtype of the saved preset.")


def load_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_video_frames():
    """Download the clip and sample frames at `VIDEO_FPS`."""
    import av

    response = requests.get(VIDEO_URL, timeout=60)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        with open(path, "wb") as f:
            f.write(response.content)
        with av.open(path) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate)
            video = np.stack(
                [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            )
    total = video.shape[0]
    count = max(4, min(int(total / source_fps * VIDEO_FPS), MAX_VIDEO_FRAMES))
    indices = np.linspace(0, total - 1, count).round().astype(int)
    return video[indices]


def chat_prompt(processor, text, media):
    """Build the prompt with the checkpoint's own chat template."""
    messages = [
        {
            "role": "user",
            "content": [{"type": media}, {"type": "text", "text": text}],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_inputs(processor, image, frames):
    """Raw inputs per case, for the HF processor and for KerasHub."""
    image_prompt = chat_prompt(processor, IMAGE_QUESTION, "image")
    video_prompt = chat_prompt(processor, VIDEO_QUESTION, "video")
    # Tell HF the frame rate and keep every frame, so it does not resample.
    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=VIDEO_FPS,
        duration=len(frames) / VIDEO_FPS,
        frames_indices=list(range(len(frames))),
    )
    return {
        "text": {
            "hf": {"text": [TEXT_PROMPT]},
            "keras": {"prompts": [TEXT_PROMPT]},
        },
        "image": {
            "hf": {"text": [image_prompt], "images": [image]},
            "keras": {"prompts": [image_prompt], "images": [np.array(image)]},
        },
        "video": {
            "hf": {
                "text": [video_prompt],
                "videos": [list(frames)],
                "video_metadata": [metadata],
                "do_sample_frames": False,
            },
            "keras": {
                "prompts": [video_prompt],
                "videos": [frames],
                "video_fps": [[VIDEO_FPS]],
            },
        },
    }


def run_hf(hf_preset, processor, inputs):
    """Run the HF model on every case, then free it."""
    print("-> Loading the HF model.")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset, device_map="cpu", dtype=torch.float32
    ).eval()
    results = {"params": sum(p.numel() for p in hf_model.parameters())}
    for name, case in inputs.items():
        print(f"   running HF: {name}")
        hf_inputs = processor(return_tensors="pt", **case["hf"])
        length = hf_inputs["input_ids"].shape[1]
        with torch.no_grad():
            logits = hf_model(**hf_inputs).logits
            generated = hf_model.generate(
                **hf_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        pixels = hf_inputs.get("pixel_values")
        if pixels is None:
            pixels = hf_inputs.get("pixel_values_videos")
        results[name] = {
            "input_ids": hf_inputs["input_ids"].numpy(),
            "pixel_values": None if pixels is None else pixels.numpy(),
            "logits": logits.float().numpy(),
            "generated": processor.decode(
                generated[0, length:], skip_special_tokens=True
            ),
        }
    del hf_model
    gc.collect()
    return results


def count_params(model):
    """Count unique parameters, so tied weights count once."""
    unique = {id(w): w for w in model.weights}.values()
    return sum(int(np.prod(w.shape)) for w in unique)


@torch.no_grad()
def compare_logits(keras_lm, inputs, hf_results, failures):
    """Compare the logits of the full model on KerasHub's own inputs.

    Every position must match within `LOGITS_ATOL`. The text and vision
    positions are also reported apart, along with a run on HF's own pixel
    values, which tells preprocessing differences from model differences.
    """
    print("\n-> Comparing token ids and logits.")
    backbone = keras_lm.backbone
    tokenizer = keras_lm.preprocessor.tokenizer
    vision_ids = [tokenizer.image_token_id, tokenizer.video_token_id]
    for name, case in inputs.items():
        hf = hf_results[name]
        preprocessed = keras_lm.preprocessor.generate_preprocess(
            case["keras"], sequence_length=hf["input_ids"].shape[1]
        )
        mask = ops.convert_to_numpy(preprocessed["padding_mask"]).astype(bool)
        keras_ids = ops.convert_to_numpy(preprocessed["token_ids"])[mask]
        if not np.array_equal(keras_ids, hf["input_ids"][0]):
            print(f"   {name}: token ids DIFFER")
            failures.append(f"{name} token ids")
            continue
        feed = {k: v for k, v in preprocessed.items() if k in backbone.input}
        logits = ops.convert_to_numpy(keras_lm(feed)).astype("float32")
        diff = np.abs(logits - hf["logits"])
        is_vision = np.isin(hf["input_ids"], vision_ids)
        max_text = diff[~is_vision].max()
        max_vision = diff[is_vision].max() if is_vision.any() else 0.0
        match = diff.max() <= LOGITS_ATOL
        print(
            f"   {name:5s} logits: mean abs diff {diff.mean():.2e}, "
            f"max abs diff {diff.max():.2e} (text positions {max_text:.2e}, "
            f"vision positions {max_vision:.2e}) "
            f"-> {'match' if match else 'DIFFERS'}"
        )
        if not match:
            failures.append(f"{name} logits")
        if hf["pixel_values"] is None:
            continue
        # HF flattens each patch as (channels, time, rows, cols). HF repeats
        # a still image over time, where KerasHub feeds a single frame.
        pixels = ops.convert_to_numpy(feed["pixel_values"]).copy()
        time, rows, cols, channels = pixels.shape[2:]
        hf_time = backbone.vision_encoder.temporal_patch_size
        hf_pixels = hf["pixel_values"].reshape(
            -1, channels, hf_time, rows, cols
        )
        hf_pixels = hf_pixels.transpose(0, 2, 3, 4, 1)[:, :time]
        pixel_diff = np.abs(pixels[0, : len(hf_pixels)] - hf_pixels).max()
        pixels[0, : len(hf_pixels)] = hf_pixels
        logits = ops.convert_to_numpy(
            keras_lm({**feed, "pixel_values": ops.convert_to_tensor(pixels)})
        ).astype("float32")
        print(
            f"          pixel values max abs diff {pixel_diff:.2e}; "
            "logits on HF pixel values: max abs diff "
            f"{np.abs(logits - hf['logits']).max():.2e}"
        )


def compare_generation(keras_lm, inputs, hf_results, failures):
    print("\n-> Comparing greedy generation.")
    keras_lm.compile(sampler="greedy")
    for name, case in inputs.items():
        hf = hf_results[name]
        output = keras_lm.generate(
            case["keras"],
            max_length=hf["input_ids"].shape[1] + MAX_NEW_TOKENS,
            strip_prompt=True,
        )
        output = output[0] if isinstance(output, list) else output
        match = output.strip() == hf["generated"].strip()
        print(f"   {name}")
        print(f"     HF       : {hf['generated']!r}")
        print(f"     KerasHub : {output!r}")
        print(f"     -> {'match' if match else 'DIFFERS'}")
        if not match:
            failures.append(f"{name} generation")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]
    processor = AutoProcessor.from_pretrained(hf_preset)
    inputs = build_inputs(processor, load_image(), load_video_frames())
    hf_results = run_hf(hf_preset, processor, inputs)

    print("\n-> Loading the KerasHub model from the HF checkpoint.")
    keras_lm = keras_hub.models.Qwen3VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    failures = []
    keras_params = count_params(keras_lm.backbone)
    print(f"\n-> KerasHub params {keras_params:,}, HF {hf_results['params']:,}")
    if keras_params != hf_results["params"]:
        failures.append("parameter count")
    compare_logits(keras_lm, inputs, hf_results, failures)
    compare_generation(keras_lm, inputs, hf_results, failures)
    if failures:
        raise SystemExit(f"\nFAILED: {', '.join(failures)}")

    # Free the checked float32 model, then convert again in `--dtype` to save.
    del keras_lm
    keras.utils.clear_session(free_memory=True)
    gc.collect()
    print(f"\n-> Saving the {FLAGS.dtype} preset to ./{preset}.")
    keras_lm = keras_hub.models.Qwen3VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype=FLAGS.dtype
    )
    keras_lm.save_to_preset(f"./{preset}")
    print("-> Preset saved.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
