"""Convert Qwen3-VL HuggingFace checkpoints to KerasHub presets.

The weights are converted by `from_preset("hf://...")`. This script validates
that model against HuggingFace on text, an image and a video: the vision
tower, every decoder layer, the logits, the pixels each preprocessor
produces, and greedy generation on the tasks the model is used for. It then
saves the preset to `./<preset>` and checks that it reloads.

CUDA is hidden unless `CUDA_VISIBLE_DEVICES` is set. HuggingFace always runs
on the CPU.

Usage:
```shell
cd tools/checkpoint_conversion
python convert_qwen3_vl_checkpoints.py --preset qwen3_vl_2b_instruct
```
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

_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
_VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
_VIDEO_FPS = 2
_MAX_NEW_TOKENS = 32
_MAX_VIDEO_FRAMES = 8
_TEXT_PROMPT = "What is Keras?"
_IMAGE_QUESTION = "Describe this image."
_VIDEO_QUESTION = "Describe this video."
# Greedy generation is compared on these tasks: name -> (media, question).
_GENERATION_TASKS = {
    "image captioning": ("image", "Describe this image in one sentence."),
    "image question": (
        "image",
        "How many cats are in this image, and what are they doing?",
    ),
    "video captioning": ("video", "Describe this video in one sentence."),
    "video question": ("video", "What is the main subject of this video?"),
}
# The tolerance is relative to the largest reference value of a component,
# so it holds at any activation scale. The logits get a wider one: float32
# rounding accumulates over every layer, and a few of the millions of logits
# of the larger models land just outside the tighter one.
_RTOL = 1e-3
_LOGITS_RTOL = 3e-3
# The pixels come from two independent resize implementations that may differ
# by one 8-bit level, which is 2 / 255 on the normalized scale. Allow two.
_PIXEL_ATOL = 4 / 255

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "Only compare pixels, components and logits, skip generation.",
)


def _to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _report_diff(
    label, keras_out, hf_out, atol, rtol, failures=None, brief=False
):
    """Print how a KerasHub output differs from the HF one.

    A mismatch is appended to `failures`. `brief` prints a single line, for
    the many decoder layers.
    """
    keras_out = _to_np(keras_out)
    hf_out = _to_np(hf_out)
    keras_shape = keras_out.shape
    if keras_out.size == hf_out.size:
        keras_out = keras_out.reshape(hf_out.shape)
    match = keras_out.shape == hf_out.shape and np.allclose(
        keras_out, hf_out, atol=atol, rtol=rtol
    )
    abs_diff = np.abs(keras_out - hf_out)
    if brief:
        print(
            f"   {label:36s} mean {abs_diff.mean():.2e}  "
            f"max {abs_diff.max():.2e}  -> {'match' if match else 'DIFFERS'}"
        )
    else:
        print(f"\n-> {label}")
        print(f"   KerasHub shape : {keras_shape}")
        print(f"   HF       shape : {hf_out.shape}")
        print(f"   KerasHub[:5]   : {keras_out.reshape(-1)[:5]}")
        print(f"   HF      [:5]   : {hf_out.reshape(-1)[:5]}")
        print(f"   mean abs diff  : {abs_diff.mean():.8f}")
        print(f"   max  abs diff  : {abs_diff.max():.8f}")
        if match:
            print(f"   -> match (atol={atol:.2e}, rtol={rtol})")
        else:
            mismatch = int(
                np.sum(~np.isclose(keras_out, hf_out, atol=atol, rtol=rtol))
            )
            print(
                f"   -> differs (atol={atol:.2e}, rtol={rtol}): "
                f"{mismatch}/{keras_out.size} elements, "
                f"{mismatch / keras_out.size * 100:.2f}%"
            )
    if not match and failures is not None:
        failures.append(label)


def _count_unique_params(model):
    unique = {id(w): w for w in model.weights}.values()
    return sum(int(np.prod(w.shape)) for w in unique)


def _load_video_frames():
    """Download a clip and sample frames at `_VIDEO_FPS` as PIL images."""
    import av

    response = requests.get(_VIDEO_URL, timeout=60)
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
    count = max(4, min(int(total / source_fps * _VIDEO_FPS), _MAX_VIDEO_FRAMES))
    indices = np.linspace(0, total - 1, count).round().astype(int)
    return [Image.fromarray(video[i]) for i in indices]


def _chat_prompt(processor, text, media=None):
    """Build the prompt with the KerasHub chat template.

    The HF template must give the same string, which keeps the two in sync.
    """
    content = [{"type": "text", "text": text}]
    if media:
        content.insert(0, {"type": media})
    messages = [{"role": "user", "content": content}]
    prompt = keras_hub.models.Qwen3_VLCausalLMPreprocessor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    hf_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prompt != hf_prompt:
        raise SystemExit(
            f"Chat template differs from HF: {prompt!r} {hf_prompt!r}"
        )
    return prompt


def _media_kwargs(kind, image, frames):
    """Processor arguments for one image or video."""
    if kind == "text":
        return {}
    if kind == "image":
        return {"images": [image]}
    # Tell HF the frame rate and keep every frame, so it does not resample.
    return {
        "videos": [frames],
        "video_metadata": [
            VideoMetadata(
                total_num_frames=len(frames),
                fps=_VIDEO_FPS,
                duration=len(frames) / _VIDEO_FPS,
                frames_indices=list(range(len(frames))),
            )
        ],
        "do_sample_frames": False,
    }


def _hf_forward(hf_model, **inputs):
    """Run HF once and keep the logits and every intermediate output."""
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda module, args, output: layer_outputs.append(
                _to_np(output[0] if isinstance(output, tuple) else output)
            )
        )
        for layer in hf_model.model.language_model.layers
    ]
    with torch.no_grad():
        logits = hf_model(**inputs).logits
        components = {}
        pixel_values = None
        for pixels, grid in (
            ("pixel_values", "image_grid_thw"),
            ("pixel_values_videos", "video_grid_thw"),
        ):
            if pixels in inputs:
                pixel_values = _to_np(inputs[pixels])
                vision = hf_model.model.visual(
                    inputs[pixels], grid_thw=inputs[grid]
                )
                components["vision"] = _to_np(vision.pooler_output)
                for i, feature in enumerate(vision.deepstack_features):
                    components[f"deepstack_{i}"] = _to_np(feature)
    for hook in hooks:
        hook.remove()
    for i, output in enumerate(layer_outputs):
        components[f"layer_{i}"] = output
    components["logits"] = _to_np(logits)
    return {
        "input_ids": inputs["input_ids"].numpy(),
        "components": components,
        "pixel_values": pixel_values,
    }


def _hf_generate(hf_model, processor, prompt, **media):
    inputs = processor(text=[prompt], return_tensors="pt", **media)
    length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = hf_model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False
        )
    return {
        "prompt": prompt,
        "length": length,
        "generated": processor.decode(
            output[0, length:], skip_special_tokens=True
        ),
    }


def _hf_reference(hf_model, hf_preset):
    """Run every HF pass, so the HF model can be freed before KerasHub's."""
    processor = AutoProcessor.from_pretrained(hf_preset)
    image = Image.open(
        BytesIO(requests.get(_IMAGE_URL, timeout=30).content)
    ).convert("RGB")
    frames = _load_video_frames()

    prompts = {
        "text": _TEXT_PROMPT,
        "image": _chat_prompt(processor, _IMAGE_QUESTION, "image"),
        "video": _chat_prompt(processor, _VIDEO_QUESTION, "video"),
    }
    reference = {}
    for kind, prompt in prompts.items():
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            **_media_kwargs(kind, image, frames),
        )
        reference[kind] = _hf_forward(hf_model, **inputs)
        reference[kind]["prompt"] = prompt
    reference["raw_image"] = np.array(image)
    reference["raw_frames"] = np.stack([np.array(f) for f in frames])

    if not FLAGS.skip_generation:
        cases = {"text completion": ("text", _TEXT_PROMPT)}
        for name, (kind, question) in _GENERATION_TASKS.items():
            cases[name] = (kind, _chat_prompt(processor, question, kind))
        reference["generation"] = {
            name: _hf_generate(
                hf_model,
                processor,
                prompt,
                **_media_kwargs(kind, image, frames),
            )
            | {"kind": kind}
            for name, (kind, prompt) in cases.items()
        }
    return reference


def _keras_components(keras_lm, preprocessed, with_layers=True):
    """Run the backbone and return the components HF is compared on."""
    backbone = keras_lm.backbone
    feed = {
        k: ops.convert_to_tensor(v)
        for k, v in preprocessed.items()
        if k in backbone.input
    }
    outputs = [backbone.output]
    if with_layers:
        outputs = [
            backbone.get_layer(f"transformer_layer_{i}").output
            for i in range(backbone.num_layers)
        ] + outputs
    probe = keras.Model(backbone.inputs, outputs)
    for tensor in probe.inputs:  # a text prompt leaves the vision inputs out
        if tensor.name not in feed:
            shape = [0 if d is None else d for d in tensor.shape]
            feed[tensor.name] = ops.zeros(shape, dtype=tensor.dtype)
    result = probe(feed)
    *layer_outputs, hidden = (
        result if isinstance(result, (list, tuple)) else [result]
    )
    components = {
        f"layer_{i}": _to_np(output) for i, output in enumerate(layer_outputs)
    }
    components["logits"] = _to_np(
        backbone.token_embedding(hidden, reverse=True)
    )
    if feed["pixel_values"].shape[1] > 0:
        features, *deepstack = backbone.vision_encoder(
            {
                "pixel_values": feed["pixel_values"],
                "image_grid_thw": feed["image_grid_thw"],
            }
        )
        components["vision"] = _to_np(features)
        for i, feature in enumerate(deepstack):
            components[f"deepstack_{i}"] = _to_np(feature)
    return components


def _compare(label, keras_lm, preprocessed, hf_ref, failures, layers=True):
    """Compare the vision tower, decoder layers and logits on one prompt."""
    print(f"\n=== {label}")
    keras_out = _keras_components(keras_lm, preprocessed, layers)
    for name, hf_out in hf_ref["components"].items():
        is_layer = name.startswith("layer_")
        rtol = _LOGITS_RTOL if name == "logits" else _RTOL
        if is_layer and not layers:
            continue
        _report_diff(
            f"{label}: {name}",
            keras_out[name],
            hf_out,
            atol=rtol * np.abs(hf_out).max(),
            rtol=rtol,
            failures=failures,
            brief=is_layer,
        )


def _hf_pixels_as_keras(keras_lm, hf_ref):
    """Reorder the HF pixel patches into the KerasHub layout."""
    vision_encoder = keras_lm.backbone.vision_encoder
    # HF packs (patches, C * T * P * P), channel first. KerasHub wants
    # (patches, T, P, P, C).
    return (
        hf_ref["pixel_values"]
        .reshape(
            -1,
            vision_encoder.in_channels,
            vision_encoder.temporal_patch_size,
            vision_encoder.patch_size,
            vision_encoder.patch_size,
        )
        .transpose(0, 2, 3, 4, 1)
    )


def _check_token_ids(preprocessed, hf_ids):
    mask = ops.convert_to_numpy(preprocessed["padding_mask"]).astype(bool)
    ids = ops.convert_to_numpy(preprocessed["token_ids"])[mask]
    np.testing.assert_array_equal(ids, hf_ids[0])


def _validate_generation(keras_lm, reference, failures):
    print("\n-> Comparing greedy generation.")
    keras_lm.compile(sampler="greedy")
    for name, case in reference["generation"].items():
        x = {"prompts": [case["prompt"]]}
        if case["kind"] == "image":
            x["images"] = [reference["raw_image"]]
        elif case["kind"] == "video":
            x["videos"] = [reference["raw_frames"]]
            x["video_fps"] = [[_VIDEO_FPS]]
        keras_text = keras_lm.generate(
            x,
            max_length=case["length"] + _MAX_NEW_TOKENS,
            strip_prompt=True,
        )
        if isinstance(keras_text, list):
            keras_text = keras_text[0]
        match = keras_text.strip() == case["generated"].strip()
        print(f"   {name}")
        print(f"     HF       : {case['generated']!r}")
        print(f"     KerasHub : {keras_text!r}")
        print(f"     -> {'match' if match else 'DIFFERS'}")
        if not match:
            failures.append(f"{name} generation")


def _validate_output(keras_lm, reference, hf_param_count):
    """Compare parameter counts, components, logits and generation.

    An image or video is compared twice. With the HF pixels the difference
    comes from the converted weights alone. With the KerasHub pixels it is
    what a user gets end to end.
    """
    failures = []
    preprocessor = keras_lm.preprocessor

    print("\n-> Comparing parameter counts.")
    keras_params = _count_unique_params(keras_lm.backbone)
    print(f"   KerasHub backbone params : {keras_params:,}")
    print(f"   HuggingFace total params : {hf_param_count:,}")
    assert keras_params == hf_param_count, "Parameter counts differ."

    ref = reference["text"]
    preprocessed = preprocessor.generate_preprocess(
        [_TEXT_PROMPT], sequence_length=ref["input_ids"].shape[1]
    )
    _check_token_ids(preprocessed, ref["input_ids"])
    _compare("text", keras_lm, preprocessed, ref, failures)

    media = {
        "image": {"images": [reference["raw_image"]]},
        "video": {
            "videos": [reference["raw_frames"]],
            "video_fps": [[_VIDEO_FPS]],
        },
    }
    for kind, kwargs in media.items():
        ref = reference[kind]
        preprocessed = preprocessor.generate_preprocess(
            {"prompts": [ref["prompt"]], **kwargs},
            sequence_length=ref["input_ids"].shape[1],
        )
        _check_token_ids(preprocessed, ref["input_ids"])
        hf_pixels = _hf_pixels_as_keras(keras_lm, ref)
        _report_diff(
            f"{kind} pixel_values (KerasHub vs HF preprocessing)",
            ops.convert_to_numpy(preprocessed["pixel_values"]),
            hf_pixels,
            atol=_PIXEL_ATOL,
            rtol=0.0,
            failures=failures,
        )
        _compare(
            f"{kind}, HF pixels",
            keras_lm,
            {
                **preprocessed,
                "pixel_values": ops.convert_to_tensor(hf_pixels[None]),
            },
            ref,
            failures,
        )
        # Reported, not gated: the resize may round a few pixels one 8-bit
        # level differently from torchvision, and the vision tower amplifies
        # that. Generation below is the end-to-end check.
        _compare(
            f"{kind}, KerasHub pixels",
            keras_lm,
            preprocessed,
            ref,
            failures=None,
            layers=False,
        )

    if not FLAGS.skip_generation:
        _validate_generation(keras_lm, reference, failures)
    return failures


def main(_):
    # Import TensorFlow only here, after torch has claimed CUDA, and keep it
    # on the CPU: it only runs tokenizer string ops.
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    print(f"\n-> Loading HF model: {hf_preset}")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset, device_map="cpu", dtype=torch.float32
    ).eval()
    hf_param_count = sum(p.numel() for p in hf_model.parameters())

    print("\n-> Running the HF reference passes.")
    reference = _hf_reference(hf_model, hf_preset)
    # Free HF so the larger variants fit next to the KerasHub model.
    del hf_model
    gc.collect()

    print("\n-> Loading KerasHub model from the HF checkpoint.")
    keras_lm = keras_hub.models.Qwen3_VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )

    failures = _validate_output(keras_lm, reference, hf_param_count)

    print(f"\n-> Saving KerasHub preset to ./{preset}.")
    keras_lm.save_to_preset(f"./{preset}")
    print("-> Preset saved.")

    # Free the converted model before reloading the preset.
    del keras_lm
    gc.collect()

    print("\n-> Verifying preset reload.")
    keras_hub.models.Qwen3_VLCausalLM.from_preset(f"./{preset}")
    print("-> Preset reload verified.")

    if failures:
        raise SystemExit(f"\nFAILED: {', '.join(failures)}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
