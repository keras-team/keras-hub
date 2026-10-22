"""Convert Qwen3-VL HuggingFace checkpoints to KerasHub presets.

The weights are converted by `from_preset("hf://...")`. This script validates
that model against HuggingFace on text, an image and a video, then saves the
preset to `./<preset>` and checks that it reloads.

The checks run in stages, and each stage loads only the weights it needs and
frees them before the next one. HF and KerasHub weights are never in memory
at the same time.

1. Preprocessing, with no model: token ids, M-RoPE positions and pixels.
2. Vision: the HF vision tower, then the KerasHub vision encoder, on the HF
   pixels.
3. Text: the HF text decoder, then a text-only KerasHub backbone, on the same
   HF token ids, positions and vision features. Every decoder layer and the
   logits are compared.
4. End to end: the full HF model for greedy generation, then the full KerasHub
   model for its logits, generation, saving and reloading.

The peak is one full float32 model, which generation and saving need. Each
stage prints the resident memory when it ends.

CUDA is hidden unless `CUDA_VISIBLE_DEVICES` is set. HuggingFace always runs
on the CPU.

Usage:
```shell
cd tools/checkpoint_conversion
python convert_qwen3_vl_checkpoints.py --preset qwen3_vl_2b_instruct
```
"""

import ctypes
import gc
import glob
import json
import os
import random
import resource
import tempfile
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import requests
import safetensors
import torch
from absl import app
from absl import flags
from huggingface_hub import snapshot_download
from keras import ops
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.video_utils import VideoMetadata

import keras_hub
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLDeepStackInject,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLInterleaveEmbeddings,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3_VLVisionEncoder,
)
from keras_hub.src.utils.transformers import convert_qwen3_vl
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

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
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/"
    "resolve/main/tennis.mp4"
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
# Pixel keys of the HF processor output, per media kind.
_HF_PIXEL_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "Only compare pixels, components and logits, skip generation.",
)


def _release(stage=None):
    """Free unreferenced memory and hand it back to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    if stage is not None:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE") / 2**30
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f"\n[memory] after {stage}: {rss:.1f} GiB, peak {peak:.1f} GiB")


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
        messages,
        add_generation_prompt=True,
        thinking="thinking" in FLAGS.preset,
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


def _hf_module(cls, config, weights_dir, prefix):
    """Build an HF module and stream in only the weights under `prefix`."""
    module = cls._from_config(config, dtype=torch.float32).eval()
    params = dict(module.named_parameters())
    loaded = set()
    for path in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
        with safetensors.safe_open(path, framework="pt") as f:
            for key in f.keys():
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix) :]
                with torch.no_grad():
                    params[name].copy_(f.get_tensor(key))
                loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise ValueError(f"No HF weights for {sorted(missing)}")
    return module


def _hf_inputs(processor, rope_model, prompt, kind, image, frames):
    """Run the HF processor and add the M-RoPE positions."""
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        **_media_kwargs(kind, image, frames),
    )
    grids = {
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
    }
    position_ids, _ = rope_model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        attention_mask=inputs["attention_mask"],
        **grids,
    )
    return inputs, position_ids


def _check_preprocessing(preprocessor, cases, failures):
    """Stage 1: compare token ids, M-RoPE positions and pixels."""
    print("\n=== Stage 1: preprocessing")
    for kind, case in cases.items():
        hf = case["hf_inputs"]
        x = {"prompts": [case["prompt"]]}
        if kind == "image":
            x["images"] = [case["raw_image"]]
        elif kind == "video":
            x["videos"] = [case["raw_frames"]]
            x["video_fps"] = [[_VIDEO_FPS]]
        if kind == "text":
            x = [case["prompt"]]
        preprocessed = preprocessor.generate_preprocess(
            x, sequence_length=hf["input_ids"].shape[1]
        )
        _check_token_ids(preprocessed, hf["input_ids"].numpy())
        keras_positions = ops.convert_to_numpy(preprocessed["position_ids"])
        _report_diff(
            f"{kind} position_ids",
            keras_positions.transpose(2, 0, 1),
            case["position_ids"],
            atol=0,
            rtol=0,
            failures=failures,
            brief=True,
        )
        if kind in _HF_PIXEL_KEYS:
            hf_pixels = case["hf_pixels"]
            # Without video, KerasHub keeps one frame per patch and the
            # vision encoder repeats it, so repeat it here to compare.
            keras_pixels = np.broadcast_to(
                ops.convert_to_numpy(preprocessed["pixel_values"]),
                (1,) + hf_pixels.shape,
            )
            _report_diff(
                f"{kind} pixel_values (KerasHub vs HF preprocessing)",
                keras_pixels,
                hf_pixels,
                atol=_PIXEL_ATOL,
                rtol=0.0,
                failures=failures,
            )
        case["keras_preprocessed"] = preprocessed


def _hf_pixels_as_keras(hf_pixels, vision_config):
    """Reorder HF pixel patches into the KerasHub layout."""
    # HF packs (patches, C * T * P * P), channel first. KerasHub wants
    # (patches, T, P, P, C).
    return (
        _to_np(hf_pixels)
        .reshape(
            -1,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        )
        .transpose(0, 2, 3, 4, 1)
    )


def _check_vision(hf_config, weights_dir, cases, failures):
    """Stage 2: the vision towers alone, on the HF pixels."""
    print("\n=== Stage 2: vision tower")
    media = {k: c for k, c in cases.items() if k in _HF_PIXEL_KEYS}

    print("\n-> Loading the HF vision tower.")
    hf_vision = _hf_module(
        modeling_qwen3_vl.Qwen3VLVisionModel,
        hf_config.vision_config,
        weights_dir,
        "model.visual.",
    )
    hf_params = sum(p.numel() for p in hf_vision.parameters())
    for kind, case in media.items():
        pixels, grid = _HF_PIXEL_KEYS[kind]
        with torch.no_grad():
            out = hf_vision(case["hf_inputs"][pixels], grid_thw=case[grid])
        features = out.pooler_output
        if isinstance(features, (list, tuple)):
            features = torch.cat(list(features), dim=0)
        case["hf_vision"] = [_to_np(features)] + [
            _to_np(f) for f in out.deepstack_features
        ]
    del hf_vision, out
    _release("the HF vision tower")

    print("\n-> Loading the KerasHub vision encoder.")
    encoder = Qwen3_VLVisionEncoder(
        **convert_qwen3_vl.convert_vision_encoder_config(
            _raw_config(weights_dir)
        )
    )
    with SafetensorLoader(weights_dir) as loader:
        convert_qwen3_vl.convert_vision_weights(encoder, loader)
    for kind, case in media.items():
        pixels = _hf_pixels_as_keras(
            case["hf_inputs"][_HF_PIXEL_KEYS[kind][0]], hf_config.vision_config
        )
        outputs = encoder(
            {
                "pixel_values": ops.convert_to_tensor(pixels[None]),
                "image_grid_thw": ops.convert_to_tensor(
                    case[_HF_PIXEL_KEYS[kind][1]].numpy()[None], "int32"
                ),
            }
        )
        names = ["vision"] + [f"deepstack_{i}" for i in range(len(outputs) - 1)]
        for name, keras_out, hf_out in zip(names, outputs, case["hf_vision"]):
            _report_diff(
                f"{kind}: {name}",
                keras_out,
                hf_out,
                atol=_RTOL * np.abs(hf_out).max(),
                rtol=_RTOL,
                failures=failures,
            )
    keras_params = _count_unique_params(encoder)
    del encoder, outputs
    keras.utils.clear_session(free_memory=True)
    _release("the KerasHub vision encoder")
    return hf_params, keras_params


def _hf_text_forward(text_model, lm_head, case, image_token_ids):
    """Run the HF decoder on the HF vision features, keep every layer."""
    input_ids = case["hf_inputs"]["input_ids"]
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda module, args, output: layer_outputs.append(
                _to_np(output[0] if isinstance(output, tuple) else output)
            )
        )
        for layer in text_model.layers
    ]
    kwargs = {}
    with torch.no_grad():
        embeds = text_model.embed_tokens(input_ids)
        if "hf_vision" in case:
            mask = torch.isin(input_ids, torch.tensor(image_token_ids))
            features, *deepstack = [
                torch.from_numpy(f) for f in case["hf_vision"]
            ]
            embeds = embeds.masked_scatter(mask[..., None], features)
            kwargs = {
                "visual_pos_masks": mask,
                "deepstack_visual_embeds": deepstack,
            }
        hidden = text_model(
            inputs_embeds=embeds,
            position_ids=case["position_ids"],
            attention_mask=case["hf_inputs"]["attention_mask"],
            use_cache=False,
            **kwargs,
        ).last_hidden_state
        logits = lm_head(hidden)
    for hook in hooks:
        hook.remove()
    components = {f"layer_{i}": o for i, o in enumerate(layer_outputs)}
    components["logits"] = _to_np(logits)
    return components


def _keras_text_forward(backbone, case, image_token_ids):
    """The text-only KerasHub backbone on the same inputs as HF."""
    input_ids = case["hf_inputs"]["input_ids"].numpy()
    token_ids = ops.convert_to_tensor(input_ids, "int32")
    padding_mask = ops.ones_like(token_ids)
    position_ids = ops.convert_to_tensor(
        np.transpose(case["position_ids"].numpy(), (1, 0, 2)), "int32"
    )
    x = backbone.token_embedding(token_ids)
    deepstack = []
    if "hf_vision" in case:
        flat = np.flatnonzero(np.isin(input_ids.reshape(-1), image_token_ids))
        vision_indices = ops.convert_to_tensor(flat[None], "int32")
        features, *deepstack = [
            ops.convert_to_tensor(f[None]) for f in case["hf_vision"]
        ]
        x = Qwen3_VLInterleaveEmbeddings(hidden_dim=backbone.hidden_dim)(
            image_embeddings=features,
            text_embeddings=x,
            vision_indices=vision_indices,
        )
        inject = Qwen3_VLDeepStackInject(hidden_dim=backbone.hidden_dim)
    rotary_embeddings = backbone.rotary_embedding.compute_embeddings(
        position_ids
    )
    components = {}
    for i, layer in enumerate(backbone.transformer_layers):
        x = layer(
            x,
            decoder_padding_mask=padding_mask,
            position_ids=position_ids,
            rotary_embeddings=rotary_embeddings,
        )
        components[f"layer_{i}"] = _to_np(x)
        if i < len(deepstack):
            x = inject(x, deepstack[i], vision_indices)
    hidden = backbone.layer_norm(x)
    components["logits"] = _to_np(
        backbone.token_embedding(hidden, reverse=True)
    )
    return components


def _report_components(label, keras_out, hf_out, failures, layers=True):
    for name, hf_value in hf_out.items():
        is_layer = name.startswith("layer_")
        if is_layer and not layers:
            continue
        rtol = _LOGITS_RTOL if name == "logits" else _RTOL
        _report_diff(
            f"{label}: {name}",
            keras_out[name],
            hf_value,
            atol=rtol * np.abs(hf_value).max(),
            rtol=rtol,
            failures=failures,
            brief=is_layer,
        )


def _check_text(hf_config, weights_dir, cases, failures):
    """Stage 3: the text decoders alone, on the HF vision features."""
    print("\n=== Stage 3: text decoder")
    text_config = hf_config.text_config
    image_token_ids = [hf_config.image_token_id, hf_config.video_token_id]
    keras_config = convert_qwen3_vl.convert_text_config(
        _raw_config(weights_dir)
    )
    tied = keras_config["tie_word_embeddings"]

    print("\n-> Loading the HF text decoder.")
    text_model = _hf_module(
        modeling_qwen3_vl.Qwen3VLTextModel,
        text_config,
        weights_dir,
        "model.language_model.",
    )
    lm_head = torch.nn.Linear(
        text_config.hidden_size, text_config.vocab_size, bias=False
    )
    if tied:
        lm_head.weight = text_model.embed_tokens.weight
    else:
        lm_head = _hf_module_weight(lm_head, weights_dir, "lm_head.weight")
    hf_params = sum(p.numel() for p in text_model.parameters())
    if not tied:
        hf_params += lm_head.weight.numel()
    for case in cases.values():
        case["hf_components"] = _hf_text_forward(
            text_model, lm_head, case, image_token_ids
        )
    del text_model, lm_head
    _release("the HF text decoder")

    print("\n-> Loading the text-only KerasHub backbone.")
    backbone = keras_hub.models.Qwen3_VLBackbone(
        **keras_config, dtype="float32"
    )
    with SafetensorLoader(weights_dir) as loader:
        convert_qwen3_vl.convert_text_weights(backbone, loader)
    for kind, case in cases.items():
        keras_out = _keras_text_forward(backbone, case, image_token_ids)
        print(f"\n-> {kind}")
        _report_components(kind, keras_out, case["hf_components"], failures)
    keras_params = _count_unique_params(backbone)
    del backbone, keras_out
    keras.utils.clear_session(free_memory=True)
    # Only the logits are needed from here on.
    for case in cases.values():
        case["hf_components"] = {"logits": case["hf_components"]["logits"]}
    _release("the KerasHub text decoder")
    return hf_params, keras_params


def _hf_module_weight(module, weights_dir, key):
    for path in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
        with safetensors.safe_open(path, framework="pt") as f:
            if key in f.keys():
                with torch.no_grad():
                    module.weight.copy_(f.get_tensor(key))
                return module
    raise ValueError(f"No HF weight {key}")


def _hf_generation(weights_dir, processor, generation_cases):
    """Stage 4a: greedy generation with the full HF model."""
    print("\n=== Stage 4: end to end")
    print("\n-> Loading the full HF model for generation.")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        weights_dir, device_map="cpu", dtype=torch.float32
    ).eval()
    results = {}
    for name, case in generation_cases.items():
        results[name] = _hf_generate(
            hf_model, processor, case["prompt"], **case["media"]
        ) | {"kind": case["kind"]}
    del hf_model
    _release("HF generation")
    return results


def _keras_logits(keras_lm, preprocessed):
    backbone = keras_lm.backbone
    feed = {
        k: ops.convert_to_tensor(v)
        for k, v in preprocessed.items()
        if k in backbone.input
    }
    hidden = backbone(feed)
    return _to_np(backbone.token_embedding(hidden, reverse=True))


def _check_end_to_end(keras_lm, cases, hf_config, failures):
    """Stage 4b: the full KerasHub model on HF and on KerasHub pixels.

    With the HF pixels the logits must match. With the KerasHub pixels they
    are only reported: the resize may round a few pixels one 8-bit level
    differently from torchvision, and the vision tower amplifies that.
    Generation is the end-to-end check.
    """
    for kind, case in cases.items():
        hf_logits = case["hf_components"]["logits"]
        preprocessed = case["keras_preprocessed"]
        runs = [("KerasHub pixels", preprocessed, None)]
        if kind in _HF_PIXEL_KEYS:
            hf_pixels = _hf_pixels_as_keras(
                case["hf_inputs"][_HF_PIXEL_KEYS[kind][0]],
                hf_config.vision_config,
            )
            runs.insert(
                0,
                (
                    "HF pixels",
                    {**preprocessed, "pixel_values": hf_pixels[None]},
                    failures,
                ),
            )
        else:
            runs = [("full model", preprocessed, failures)]
        for label, feed, gate in runs:
            _report_diff(
                f"{kind}, {label}: logits",
                _keras_logits(keras_lm, feed),
                hf_logits,
                atol=_LOGITS_RTOL * np.abs(hf_logits).max(),
                rtol=_LOGITS_RTOL,
                failures=gate,
            )


def _raw_config(weights_dir):
    """The HF `config.json` as KerasHub's converter reads it."""
    with open(os.path.join(weights_dir, "config.json")) as f:
        return json.load(f)


def _download(hf_preset):
    """Return a local folder with the HF config, processor and weights."""
    if os.path.isdir(hf_preset):
        return hf_preset
    return snapshot_download(
        hf_preset,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )


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
    # KerasHub converts from the same local folder as HF.
    weights_dir = _download(hf_preset)
    hf_config = AutoConfig.from_pretrained(weights_dir)
    processor = AutoProcessor.from_pretrained(weights_dir)
    preprocessor = keras_hub.models.Qwen3_VLCausalLMPreprocessor.from_preset(
        weights_dir
    )
    # `get_rope_index` only reads the config, so a weightless model will do.
    with torch.device("meta"):
        rope_model = modeling_qwen3_vl.Qwen3VLModel(hf_config)
    failures = []

    image = Image.open(
        BytesIO(requests.get(_IMAGE_URL, timeout=30).content)
    ).convert("RGB")
    frames = _load_video_frames()
    raw_image = np.array(image)
    raw_frames = np.stack([np.array(f) for f in frames])
    cases = {}
    for kind, question in (
        ("text", _TEXT_PROMPT),
        ("image", _IMAGE_QUESTION),
        ("video", _VIDEO_QUESTION),
    ):
        prompt = (
            question
            if kind == "text"
            else _chat_prompt(processor, question, kind)
        )
        hf_inputs, position_ids = _hf_inputs(
            processor, rope_model, prompt, kind, image, frames
        )
        case = {
            "prompt": prompt,
            "hf_inputs": hf_inputs,
            "position_ids": position_ids,
            "raw_image": raw_image,
            "raw_frames": raw_frames,
        }
        if kind in _HF_PIXEL_KEYS:
            pixels, grid = _HF_PIXEL_KEYS[kind]
            case["hf_pixels"] = _hf_pixels_as_keras(
                hf_inputs[pixels], hf_config.vision_config
            )
            case[grid] = hf_inputs[grid]
        cases[kind] = case

    _check_preprocessing(preprocessor, cases, failures)
    hf_vision_params, keras_vision_params = _check_vision(
        hf_config, weights_dir, cases, failures
    )
    hf_text_params, keras_text_params = _check_text(
        hf_config, weights_dir, cases, failures
    )
    hf_params = hf_vision_params + hf_text_params
    keras_params = keras_vision_params + keras_text_params
    print(f"\n-> KerasHub params {keras_params:,}, HF params {hf_params:,}")
    if keras_params != hf_params:
        failures.append("parameter count")

    hf_generation = None
    if not FLAGS.skip_generation:
        generation_cases = {
            "text completion": {
                "kind": "text",
                "prompt": _TEXT_PROMPT,
                "media": {},
            }
        }
        for name, (kind, question) in _GENERATION_TASKS.items():
            generation_cases[name] = {
                "kind": kind,
                "prompt": _chat_prompt(processor, question, kind),
                "media": _media_kwargs(kind, image, frames),
            }
        hf_generation = _hf_generation(weights_dir, processor, generation_cases)

    print("\n-> Loading the full KerasHub model from the HF checkpoint.")
    keras_lm = keras_hub.models.Qwen3_VLCausalLM.from_preset(
        weights_dir, dtype="float32"
    )
    full_params = _count_unique_params(keras_lm.backbone)
    print(f"   full KerasHub backbone params: {full_params:,}")
    if full_params != hf_params:
        failures.append("full model parameter count")
    _check_end_to_end(keras_lm, cases, hf_config, failures)
    if hf_generation is not None:
        _validate_generation(
            keras_lm,
            {
                "generation": hf_generation,
                "raw_image": raw_image,
                "raw_frames": raw_frames,
            },
            failures,
        )

    print(f"\n-> Saving KerasHub preset to ./{preset}.")
    keras_lm.save_to_preset(f"./{preset}")
    print("-> Preset saved.")
    del keras_lm
    keras.utils.clear_session(free_memory=True)
    _release("the full KerasHub model")

    print("\n-> Verifying preset reload.")
    keras_hub.models.Qwen3_VLCausalLM.from_preset(f"./{preset}")
    print("-> Preset reload verified.")
    _release("the reload")

    if failures:
        raise SystemExit(f"\nFAILED: {', '.join(failures)}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
