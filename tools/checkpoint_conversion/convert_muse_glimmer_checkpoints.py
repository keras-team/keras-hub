"""Convert Muse Glimmer HuggingFace checkpoints to KerasHub preset format.

Also handles the `-assistant` DFlash speculative-decoding drafter preset
(`muse_glimmer_30b_assistant`).

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
from transformers import AutoModel
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
    "muse_glimmer_30b": "meta-models/Muse-Glimmer-30B",
    "muse_glimmer_30b_assistant": "meta-models/Muse-Glimmer-30B-assistant",
}

# The assistant/drafter has no vocabulary or tokenizer of its own — it
# consumes `noise_embeds`/`context_hidden_states` built from the separate
# target model. A short random target-length sequence is enough to
# exercise every weight and mask branch (block-diffusion window + cross-
# model context).
ASSISTANT_TARGET_SEQUENCE_LENGTH = 24

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
ASSET_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "keras_hub_muse_glimmer"
)

TEXT_PROMPT = "What is Keras?"
# Bare placeholders. The HF processor and the KerasHub preprocessor both
# expand them to the full layout with the delimiter tokens.
IMAGE_PROMPT = "<|patch|>Describe this image."
VIDEO_PROMPT = "<|video|>Describe this video."

MAX_NEW_TOKENS = 24

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


def _fetch_asset(url, timeout):
    """Return the bytes at `url`, cached on disk after the first download.

    Some hosts return 403 to the default python-requests user agent.
    """
    cache_path = os.path.join(ASSET_CACHE_DIR, os.path.basename(url))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
    )
    response.raise_for_status()
    os.makedirs(ASSET_CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(response.content)
    return response.content


def _load_image_asset():
    return Image.open(BytesIO(_fetch_asset(IMAGE_URL, 30))).convert("RGB")


def _load_video_asset():
    """Download, decode, and subsample the test video's frames.

    Pinned to a small frame count. This keeps `t > 1` after temporal
    patching, while bounding peak memory for the O(n^2) attention pass.
    """
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "PyAV is required for Muse Glimmer video validation."
        ) from error

    try:
        container = av.open(BytesIO(_fetch_asset(VIDEO_URL, 60)))
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)
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
    frames = np.stack(frames)
    total_num_frames = frames.shape[0]
    num_frames = 2
    if total_num_frames > num_frames:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames)
        indices = indices.astype(int)[:num_frames]
    else:
        indices = np.arange(total_num_frames)
    video_metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=fps,
        height=frames.shape[1],
        width=frames.shape[2],
        frames_indices=indices.tolist(),
    )
    return frames[indices], video_metadata


def _load_test_assets():
    """Load the image/video assets shared by every preset flow."""
    raw_image = _load_image_asset()
    raw_video, video_metadata = _load_video_asset()
    return raw_image, raw_video, video_metadata


def _count_keras_params(backbone):
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


def _strip_chat_markers(text):
    """Strip literal chat-format tokens KerasHub's decode doesn't skip."""
    for marker in ("<|message|>", "<|eom|>", "<|start|>", "<|eot|>"):
        text = text.replace(marker, "")
    return text


def _build_hf_multimodal_inputs(
    hf_tokenizer, processor, prompt, media, modality, video_metadata=None
):
    """Build HF generate() inputs for an image/video prompt.

    The HF processor expands the placeholder to the full layout, with the
    delimiter tokens and, for a video, the `Time: ...s` text. The tool
    passes the frames that `_load_video_asset` sampled, so it turns off
    the frame sampling of the processor.
    """
    if modality == "image":
        processor_outputs = processor(
            text=[prompt], images=[media], return_tensors="pt"
        )
        keys = ("input_ids", "attention_mask", "pixel_values")
        keys += ("image_grid_thw",)
    else:
        processor_outputs = processor(
            text=[prompt],
            videos=[media],
            video_metadata=[video_metadata],
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )
        keys = ("input_ids", "attention_mask", "pixel_values_videos")
        keys += ("video_grid_thw",)
    return {key: processor_outputs[key] for key in keys}


def _precompute_multimodal_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    prompt,
    media,
    modality,
    video_metadata=None,
):
    hf_inputs = _build_hf_multimodal_inputs(
        hf_tokenizer,
        processor,
        prompt,
        media,
        modality,
        video_metadata=video_metadata,
    )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, use_cache=False)

    result = {
        "prompt": prompt,
        "input_ids": hf_inputs["input_ids"].cpu().numpy().astype(np.int32),
        "logits": hf_outputs.logits.detach().cpu().float().numpy(),
        "media": np.asarray(media) if modality == "image" else media,
    }

    if not FLAGS.skip_generation:
        with torch.no_grad():
            hf_generated = hf_model.generate(
                **hf_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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

    processor = AutoProcessor.from_pretrained(hf_preset)
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        results["text_generated"] = hf_tokenizer.decode(
            hf_gen[0, hf_ids.shape[1] :], skip_special_tokens=True
        )

    raw_image, raw_video, video_metadata = _load_test_assets()
    results["image"] = _precompute_multimodal_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        IMAGE_PROMPT,
        raw_image,
        "image",
    )

    results["video"] = _precompute_multimodal_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        VIDEO_PROMPT,
        raw_video,
        "video",
        video_metadata=video_metadata,
    )

    return results


def test_parameter_count(keras_backbone, hf_param_count):
    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")
    np.testing.assert_equal(keras_params, hf_param_count)
    print("  ✓ Parameter counts match!")


def test_token_ids(keras_model, hf_results, label):
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
    print(f"\nHF token IDs: {hf_ids[0][:10].tolist()}")
    print(f"KH token IDs: {keras_valid[:10].tolist()}")
    np.testing.assert_array_equal(keras_valid, hf_ids[0])
    print(f" ✓ [{label}] Token IDs match.")


def _report_numerics(label, keras_logits, hf_logits, vision_indices):
    """Print the logit gap and return `True` if the check fails.

    The check requires `atol=1e-3, rtol=1e-3` on every position that is not
    a vision position. The vision positions only print their gap.
    """
    abs_diff = np.abs(keras_logits - hf_logits)
    per_position = abs_diff.reshape(-1, abs_diff.shape[-1]).max(axis=-1)
    print(f"\nlogit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"logit max absolute diff:  {abs_diff.max():.6f}")
    print(
        f"logit max diff at worst position ({int(per_position.argmax())} of "
        f"{per_position.size}): {per_position.max():.6f}"
    )
    print(f"logit max diff at last position: {per_position[-1]:.6f}")
    is_vision = np.zeros(per_position.shape, dtype=bool)
    if vision_indices is not None:
        is_vision[np.asarray(vision_indices).reshape(-1)] = True
        vision_diff = per_position[is_vision]
        print(
            f"logit max diff at vision positions: {vision_diff.max():.6f} "
            f"(mean {vision_diff.mean():.6f})"
        )
    tol = 1e-3 + 1e-3 * np.abs(hf_logits)
    outside = (abs_diff > tol).reshape(-1, abs_diff.shape[-1]).any(axis=-1)
    failed_positions = int(np.sum(outside & ~is_vision))
    checked_positions = int(np.sum(~is_vision))
    print(
        f"  [{label}] {failed_positions}/{checked_positions} non-vision "
        "positions have logits outside atol=1e-3, rtol=1e-3."
    )
    if failed_positions:
        print(f" ✗ [{label}] logits differ from HF.")
    else:
        print(f" ✓ [{label}] logits match HF on non-vision positions.")
    return failed_positions > 0


def _prepare_media_inputs(keras_model, result, media_key):
    """Run a raw prompt and its media through the preprocessor.

    Returns the batch-of-one inputs that the model expects.
    """
    preprocessed = keras_model.preprocessor.generate_preprocess(
        {"prompts": [result["prompt"]], media_key: result["media"]},
        sequence_length=result["input_ids"].shape[1],
    )
    preprocessed = {
        key: ops.convert_to_numpy(value) for key, value in preprocessed.items()
    }
    return {
        "token_ids": preprocessed["token_ids"].reshape(1, -1),
        "padding_mask": preprocessed["padding_mask"].reshape(1, -1),
        "pixel_values": preprocessed["pixel_values"][np.newaxis],
        "image_grid_thw": preprocessed["image_grid_thw"][np.newaxis],
        "vision_indices": preprocessed["vision_indices"].reshape(1, -1),
    }


def test_numerics(keras_model, hf_results, label):
    """Compare logits with HF and return `True` if the check fails.

    Every check runs the raw prompt and media through the preprocessor.
    The image and video converters are in the measured pass.
    """
    if label == "TEXT":
        model_inputs, _, _ = keras_model.preprocessor(
            [TEXT_PROMPT],
            sequence_length=hf_results["text_token_ids"].shape[1],
        )
        with torch.no_grad():
            keras_logits = ops.convert_to_numpy(
                keras_model(model_inputs)
            ).astype(np.float32)
        return _report_numerics(
            label, keras_logits, hf_results["text_logits"], None
        )

    result = hf_results[label.lower()]
    media_key = "images" if label == "IMAGE" else "videos"
    model_inputs = _prepare_media_inputs(keras_model, result, media_key)
    with torch.no_grad():
        keras_logits = ops.convert_to_numpy(keras_model(model_inputs)).astype(
            np.float32
        )
    return _report_numerics(
        label,
        keras_logits,
        result["logits"],
        model_inputs["vision_indices"][0],
    )


def test_generation(keras_model, hf_results, label):
    if label == "TEXT":
        max_length = hf_results["text_token_ids"].shape[1] + MAX_NEW_TOKENS
        keras_output = keras_model.generate(
            TEXT_PROMPT, max_length=max_length, strip_prompt=True
        )
        hf_output = hf_results.get("text_generated", "N/A")
    else:
        result = hf_results[label.lower()]
        media_key = "images" if label == "IMAGE" else "videos"
        keras_output = keras_model.generate(
            {
                "prompts": [result["prompt"]],
                media_key: result["media"],
            },
            max_length=result["input_ids"].shape[1] + MAX_NEW_TOKENS,
            strip_prompt=True,
        )
        if isinstance(keras_output, (list, tuple)):
            keras_output = keras_output[0]
        hf_output = result.get("generated", "N/A")

    keras_output = _strip_chat_markers(keras_output)

    print(f"\n  {label} KerasHub: {keras_output}")
    print(f"  {label} HF:       {hf_output}")
    print(f" ✓ [{label}] Text generation completed.")


def validate_output(keras_model, hf_results):
    print("\n--- Parameter Count ---")
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])

    labels = ("TEXT", "IMAGE", "VIDEO")

    print("\n--- Token ID Verification ---")
    for label in labels:
        test_token_ids(keras_model, hf_results, label)

    print("\n--- Numerics Verification ---")
    failed = [
        label
        for label in labels
        if test_numerics(keras_model, hf_results, label)
    ]

    if not FLAGS.skip_generation:
        keras_model.compile(sampler="greedy")

        print("\n--- Text Generation ---")
        for label in labels:
            test_generation(keras_model, hf_results, label)

    if failed:
        raise AssertionError(
            f"Logits differ from HF beyond tolerance for: {failed}."
        )


def save_preset(keras_model, preset_name):
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


def _load_hf_assistant_models(hf_preset):
    """The target model id is the assistant id with `-assistant` removed."""
    target_preset = hf_preset.replace("-assistant", "")
    hf_target_model = AutoModelForImageTextToText.from_pretrained(
        target_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        force_download=False,
    )
    hf_target_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        target_preset, force_download=False
    )
    processor = AutoProcessor.from_pretrained(target_preset)
    hf_assistant_model = AutoModel.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        force_download=False,
    )
    hf_assistant_model.eval()
    print("-> HuggingFace target + assistant models loaded.")
    return hf_target_model, hf_tokenizer, processor, hf_assistant_model


def _build_assistant_target_context(
    hf_target_model, target_layer_ids, hidden_dim
):
    """Run the target model on a random token sequence and concatenate its
    hidden states at `target_layer_ids` into `context_hidden_states`.

    Returns:
        A tuple `(input_ids, context_hidden_states)`, the latter shaped
        `(1, sequence_length, len(target_layer_ids) * hidden_dim)`.
    """
    vocab_size = hf_target_model.config.get_text_config().vocab_size
    input_ids = torch.randint(
        0,
        vocab_size,
        (1, ASSISTANT_TARGET_SEQUENCE_LENGTH),
        dtype=torch.long,
    )
    # Match the DFlash candidate generator. A list `output_hidden_states`
    # has no embedding entry, so id `i` is the output of decoder layer `i`,
    # and the last layer's entry is the final-norm output.
    with torch.no_grad():
        target_out = hf_target_model(
            input_ids=input_ids, output_hidden_states=list(target_layer_ids)
        )
    selected = [target_out.hidden_states[i] for i in target_layer_ids]
    context_hidden_states = torch.cat(selected, dim=-1)
    assert context_hidden_states.shape[-1] == len(target_layer_ids) * hidden_dim
    return input_ids, context_hidden_states


def _precompute_assistant_multimodal_outputs(
    hf_target_model,
    hf_assistant_model,
    hf_tokenizer,
    processor,
    prompt,
    media,
    modality,
    video_metadata=None,
):
    hf_inputs = _build_hf_multimodal_inputs(
        hf_tokenizer, processor, prompt, media, modality, video_metadata
    )
    media_key = "images" if modality == "image" else "videos"

    with torch.no_grad():
        hf_spec_ids = hf_target_model.generate(
            **hf_inputs,
            assistant_model=hf_assistant_model,
            speculation_type="dflash",
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    prompt_length = hf_inputs["input_ids"].shape[1]
    return {
        "hf_generated": hf_tokenizer.decode(
            hf_spec_ids[0, prompt_length:], skip_special_tokens=True
        ),
        "kh_inputs": {"prompts": [prompt], media_key: media},
        "input_ids": hf_inputs["input_ids"],
    }


def _precompute_assistant_hf_outputs(
    hf_target_model, hf_assistant_model, hf_tokenizer, processor
):
    """Precompute HF speculative-generation outputs for text/image/video."""
    text_prompt = TEXT_PROMPT
    hf_text_inputs = hf_tokenizer(text_prompt, return_tensors="pt")
    with torch.no_grad():
        hf_spec_ids = hf_target_model.generate(
            **hf_text_inputs,
            assistant_model=hf_assistant_model,
            speculation_type="dflash",
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    text_prompt_length = hf_text_inputs["input_ids"].shape[1]
    text_result = {
        "hf_generated": hf_tokenizer.decode(
            hf_spec_ids[0, text_prompt_length:], skip_special_tokens=True
        ),
        "kh_inputs": text_prompt,
        "input_ids": hf_text_inputs["input_ids"],
    }

    raw_image, raw_video, video_metadata = _load_test_assets()

    return {
        "TEXT": text_result,
        "IMAGE": _precompute_assistant_multimodal_outputs(
            hf_target_model,
            hf_assistant_model,
            hf_tokenizer,
            processor,
            IMAGE_PROMPT,
            raw_image,
            "image",
        ),
        "VIDEO": _precompute_assistant_multimodal_outputs(
            hf_target_model,
            hf_assistant_model,
            hf_tokenizer,
            processor,
            VIDEO_PROMPT,
            raw_video,
            "video",
            video_metadata=video_metadata,
        ),
    }


def test_assistant_generation(
    target_preset,
    kh_assistant,
    hf_gen_data,
    input_ids,
    context_hidden_states,
    target_layer_ids,
):
    print("-> Loading KerasHub target model...")
    kh_target = keras_hub.models.MuseGlimmerCausalLM.from_preset(
        f"hf://{target_preset}", dtype="float32"
    )
    kh_target.compile(sampler="greedy")
    print("\n--- Section 3: Target context ---")
    # The Keras target must select the same layers as the HF generator.
    _, _, kh_context = kh_target._build_cache(
        input_ids.numpy().astype(np.int32),
        ops.ones(tuple(input_ids.shape), dtype="int32"),
        target_layer_ids=target_layer_ids,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(kh_context),
        context_hidden_states.numpy(),
        atol=1e-3,
        err_msg="Target context differs from the HF generator context.",
    )
    print("✓ Target context matches the HF generator context.")
    print("\n--- Section 4: Speculative generation ---")
    for label, data in hf_gen_data.items():
        kh_output = kh_target.generate(
            data["kh_inputs"],
            max_length=data["input_ids"].shape[1] + MAX_NEW_TOKENS,
            strip_prompt=True,
            assistant_model=kh_assistant,
        )
        if isinstance(kh_output, (list, tuple)):
            kh_output = kh_output[0]
        kh_output = _strip_chat_markers(kh_output)
        print(f"\n  [{label}] HF speculative:  {data['hf_generated']}")
        print(f"  [{label}] KH speculative:  {kh_output}")

    del kh_target
    gc.collect()


def verify_assistant_mode(preset, hf_preset):
    target_preset = hf_preset.replace("-assistant", "")
    hf_target_model, hf_tokenizer, processor, hf_assistant_model = (
        _load_hf_assistant_models(hf_preset)
    )
    config = hf_assistant_model.config
    target_layer_ids = config.target_layer_ids
    hidden_dim = config.hidden_size
    block_size = getattr(config, "block_size", 16)

    print("-> Building target context from a random token sequence...")
    input_ids, context_hidden_states = _build_assistant_target_context(
        hf_target_model, target_layer_ids, hidden_dim
    )
    noise_embeds = torch.randn(1, block_size, hidden_dim)

    print("-> Running HF assistant forward pass...")
    with torch.no_grad():
        hf_out = hf_assistant_model(
            noise_embeds=noise_embeds,
            context_hidden_states=context_hidden_states,
        )
    hf_hidden_states = hf_out.last_hidden_state.detach().cpu().numpy()
    hf_params = sum(p.numel() for p in hf_assistant_model.parameters())

    hf_gen_data = None
    if not FLAGS.skip_generation:
        print("-> Precomputing HF speculative-generation outputs...")
        hf_gen_data = _precompute_assistant_hf_outputs(
            hf_target_model, hf_assistant_model, hf_tokenizer, processor
        )

    del hf_target_model, hf_assistant_model
    gc.collect()

    print("-> Loading KerasHub model...")
    kh_assistant = keras_hub.models.MuseGlimmerAssistantCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )

    print("\n--- Section 1: Parameter count ---")
    kh_params = _count_keras_params(kh_assistant.backbone)
    print(f"   KH params: {kh_params:,}")
    print(f"   HF params: {hf_params:,}")
    np.testing.assert_equal(kh_params, hf_params)
    print("✓ Parameter counts match.")

    print("\n--- Section 2: Logit numerics ---")
    # call_with_cache requires a real cache; write the whole synthetic
    # context sequence in one cache_update_index=0 call.
    assistant_cache = ops.zeros(
        [
            1,
            config.num_hidden_layers,
            2,
            ASSISTANT_TARGET_SEQUENCE_LENGTH,
            config.num_key_value_heads,
            config.head_dim,
        ],
        dtype="float32",
    )
    kh_out, _ = kh_assistant.call_with_cache(
        noise_embeds=noise_embeds.numpy(),
        context_hidden_states=context_hidden_states.numpy(),
        cache=assistant_cache,
        cache_update_index=0,
    )
    kh_hidden_states = ops.convert_to_numpy(kh_out)

    abs_diff = np.abs(kh_hidden_states - hf_hidden_states)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"   max |Δ| = {max_diff:.6f},  mean |Δ| = {mean_diff:.6f}")
    np.testing.assert_allclose(
        kh_hidden_states,
        hf_hidden_states,
        atol=1e-3,
        rtol=1e-3,
        err_msg="Assistant hidden states differ from HF beyond tolerance.",
    )
    print("✓ Hidden states within tolerance (atol=1e-3, rtol=1e-3).")

    if not FLAGS.skip_generation:
        test_assistant_generation(
            target_preset,
            kh_assistant,
            hf_gen_data,
            input_ids,
            context_hidden_states,
            target_layer_ids,
        )

    del kh_assistant
    gc.collect()
    kh_save = keras_hub.models.MuseGlimmerAssistantCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    save_preset(kh_save, preset)


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    if "assistant" in preset:
        verify_assistant_mode(preset, hf_preset)
        return

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

    validate_output(keras_model, hf_results)

    # Parity was just verified in float32; always save in bfloat16.
    preprocessor_ref = keras_model.preprocessor
    del keras_model
    gc.collect()
    backbone_bf16 = keras_hub.models.MuseGlimmerBackbone.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model_bf16 = keras_hub.models.MuseGlimmerCausalLM(
        backbone=backbone_bf16, preprocessor=preprocessor_ref
    )
    save_preset(keras_model_bf16, preset)
    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
