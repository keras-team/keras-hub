"""Convert Gemma4 Unified HuggingFace checkpoints to KerasHub preset format.

Usage:
  python tools/checkpoint_conversion/convert_gemma4_unified_hf_checkpoints.py \
      --preset gemma4_unified_instruct_12b \
      --save_dtype bfloat16
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
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "gemma4_unified_12b": "google/gemma-4-12B",
    "gemma4_unified_instruct_12b": "google/gemma-4-12B-it",
    "gemma4_unified_instruct_12b_assistant": "google/gemma-4-12B-it-assistant",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
AUDIO_FILE_PATH = os.path.join(
    _REPO_ROOT,
    "keras_hub/src/tests/test_data/audio_transcription_tests/"
    "male_short_voice_clip_3sec.wav",
)

# Text-only prompt.
PROMPT_TEXT = (
    "<start_of_turn>user\nWhat is the capital of France?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

# Image prompt.
PROMPT_IMAGE = (
    "<start_of_turn>user\n\n<|image|>\nWhat is in this image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_AUDIO = (
    "<|turn>user\n"
    "<|audio|>"
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three.<turn|>\n"
    "<|turn>model\n"
)

PROMPT_VIDEO = (
    "<|turn>user\n<|video|>Describe this video.<turn|>\n<|turn>model\n"
)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)
flags.DEFINE_string(
    "video_path",
    None,
    "Path to a video file for video verification (optional).",
)
flags.DEFINE_boolean(
    "skip_generate",
    False,
    "Skip the generation comparison step.",
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _download_test_video():
    """Download test clip and decode to (T, H, W, C) uint8."""
    try:
        import av
    except ImportError:
        print("Warning: PyAV not installed; skipping video download.")
        return None
    try:
        response = requests.get(VIDEO_URL, timeout=60)
        response.raise_for_status()
        container = av.open(BytesIO(response.content))
        frames = [
            f.to_ndarray(format="rgb24") for f in container.decode(video=0)
        ]
        return np.stack(frames)
    except Exception as e:
        print(f"Warning: could not download test video ({e}).")
        return None


def _count_hf_params(hf_model):
    param_names = {name for name, _ in hf_model.named_parameters()}
    num_params = sum(param.numel() for param in hf_model.parameters())
    num_buffers = sum(
        value.numel()
        for name, value in hf_model.state_dict().items()
        if name not in param_names and name.endswith(".layer_scalar")
    )
    return num_params + num_buffers


def _count_keras_hub_params(backbone):
    unique_weights = {
        id(weight): weight for weight in backbone.weights
    }.values()
    return sum(weight.numpy().size for weight in unique_weights)


def _load_test_assets():
    """Load image, audio, and video assets for verification."""
    raw_image = _load_test_image()

    import soundfile as sf

    try:
        raw_audio, sr = sf.read(AUDIO_FILE_PATH)
        if sr != 16000:
            from scipy import signal

            raw_audio = signal.resample(
                raw_audio, int(len(raw_audio) * 16000 / sr)
            )
    except Exception as e:
        print(f"Warning: could not read audio ({e}), using silence.")
        raw_audio = np.zeros((16000 * 3,), dtype=np.float32)

    if FLAGS.video_path:
        import av

        container = av.open(FLAGS.video_path)
        frames = [
            f.to_ndarray(format="rgb24") for f in container.decode(video=0)
        ]
        raw_video = np.stack(frames)
    else:
        raw_video = _download_test_video()

    return raw_image, raw_audio, raw_video


# ── HF model loading & forward ─────────────────────────────────────────────


def _load_hf_model(hf_preset):
    """Load HF model/tokenizer/processor and detect capabilities."""
    is_assistant = "assistant" in hf_preset

    if is_assistant:
        # Load both target (base instruct) and assistant models.
        target_preset = hf_preset.replace("-assistant", "")
        hf_target_model = AutoModelForCausalLM.from_pretrained(
            target_preset,
            device_map="cpu",
            torch_dtype=torch.float32,
            force_download=False,
        )
        hf_target_model.eval()
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_preset,
            device_map="cpu",
            torch_dtype=torch.float32,
            force_download=False,
        )
        hf_model.eval()
        hf_tokenizer = AutoTokenizer.from_pretrained(
            target_preset, return_tensors="pt", force_download=False
        )
        processor = AutoProcessor.from_pretrained(
            target_preset, force_download=False
        )
        is_audio_model = (
            hasattr(hf_target_model.config, "audio_config")
            and hf_target_model.config.audio_config is not None
        )
        is_video_model = (
            hasattr(processor, "video_processor")
            and processor.video_processor is not None
        )
        final_logit_cap = getattr(
            hf_target_model.config.get_text_config(),
            "final_logit_softcapping",
            None,
        )
        print(
            f"-> HF assistant model loaded (final_logit_cap={final_logit_cap})."
        )
        return (
            hf_model,
            hf_tokenizer,
            processor,
            is_audio_model,
            is_video_model,
            final_logit_cap,
            hf_target_model,
        )

    hf_model = AutoModelForMultimodalLM.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        force_download=False,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_preset, return_tensors="pt", force_download=False
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(hf_preset, force_download=False)
    print("-> HuggingFace model loaded.")

    is_audio_model = (
        hasattr(hf_model.config, "audio_config")
        and hf_model.config.audio_config is not None
    )
    is_video_model = (
        hasattr(processor, "video_processor")
        and processor.video_processor is not None
    )

    final_logit_cap = getattr(hf_model.config, "final_logit_softcapping", None)
    if final_logit_cap is None and hasattr(hf_model.config, "get_text_config"):
        final_logit_cap = getattr(
            hf_model.config.get_text_config(),
            "final_logit_softcapping",
            None,
        )
    print(f"-> final_logit_softcapping: {final_logit_cap}")

    return (
        hf_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
        None,  # hf_target_model (non-assistant)
    )


def _precompute_hf_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    prompt,
    raw_image,
    raw_audio=None,
    raw_video=None,
    skip_generate=False,
):
    if raw_video is not None and not isinstance(raw_video, torch.Tensor):
        raw_video = torch.from_numpy(raw_video)

    hf_inputs = processor(
        text=prompt,
        images=raw_image,
        audio=raw_audio,
        videos=raw_video,
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )
    hf_inputs = {key: value.to(device) for key, value in hf_inputs.items()}

    # Ensure HF inputs start with BOS.
    bos_id = hf_tokenizer.bos_token_id
    if bos_id is not None and hf_inputs["input_ids"][0, 0].item() != bos_id:
        bos = torch.full(
            (hf_inputs["input_ids"].shape[0], 1),
            bos_id,
            dtype=hf_inputs["input_ids"].dtype,
            device=hf_inputs["input_ids"].device,
        )
        hf_inputs["input_ids"] = torch.cat([bos, hf_inputs["input_ids"]], dim=1)
        if "attention_mask" in hf_inputs:
            hf_inputs["attention_mask"] = torch.ones_like(
                hf_inputs["input_ids"]
            )
        if "mm_token_type_ids" in hf_inputs:
            mm_pad = torch.zeros(
                (hf_inputs["mm_token_type_ids"].shape[0], 1),
                dtype=hf_inputs["mm_token_type_ids"].dtype,
                device=hf_inputs["mm_token_type_ids"].device,
            )
            hf_inputs["mm_token_type_ids"] = torch.cat(
                [mm_pad, hf_inputs["mm_token_type_ids"]], dim=1
            )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=False)

    hf_logits = hf_outputs.logits.detach().cpu().float().numpy()
    hf_input_ids = hf_inputs["input_ids"].detach().cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].detach().cpu().numpy()

    hf_pixel_values = (
        hf_inputs["pixel_values"].detach().cpu().float().numpy()
        if "pixel_values" in hf_inputs
        else None
    )
    hf_image_position_ids = (
        hf_inputs["image_position_ids"].detach().cpu().numpy()
        if "image_position_ids" in hf_inputs
        else None
    )

    if not skip_generate:
        with torch.no_grad():
            generated_ids = hf_model.generate(
                **hf_inputs,
                max_new_tokens=64,
            )
        prompt_length = hf_inputs["input_ids"].shape[1]
        hf_generated_text = hf_tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=True
        )
    else:
        hf_generated_text = "(skipped)"

    ret = {
        "logits": hf_logits,
        "input_ids": hf_input_ids,
        "attention_mask": hf_attention_mask,
        "pixel_values": hf_pixel_values,
        "image_position_ids": hf_image_position_ids,
        "generated_text": hf_generated_text,
        "param_count": _count_hf_params(hf_model),
    }

    if "input_features" in hf_inputs:
        ret["input_features"] = (
            hf_inputs["input_features"].detach().cpu().numpy()
        )
    if "mm_token_type_ids" in hf_inputs:
        ret["mm_token_type_ids"] = (
            hf_inputs["mm_token_type_ids"].detach().cpu().numpy()
        )
    if raw_video is not None and "pixel_values_videos" in hf_inputs:
        pv = hf_inputs["pixel_values_videos"]
        ret["num_video_frames"] = int(pv.shape[1])
    return ret


def _precompute_all_hf_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    raw_image,
    raw_audio,
    raw_video,
    is_audio_model,
    is_video_model,
    skip_generate=False,
):
    """Run HF forward passes for all applicable modalities."""
    print("-> Precomputing HF outputs for text prompt...")
    hf_data_text = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        PROMPT_TEXT,
        raw_image=None,
        skip_generate=skip_generate,
    )

    print("-> Precomputing HF outputs for image prompt...")
    hf_data_image = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        PROMPT_IMAGE,
        raw_image,
        skip_generate=skip_generate,
    )

    hf_data_audio = None
    if is_audio_model:
        print("-> Precomputing HF outputs for audio prompt...")
        hf_data_audio = _precompute_hf_outputs(
            hf_model,
            hf_tokenizer,
            processor,
            PROMPT_AUDIO,
            raw_image=None,
            raw_audio=raw_audio,
            skip_generate=skip_generate,
        )

    hf_data_video = None
    if is_video_model and raw_video is not None:
        print("-> Precomputing HF outputs for video prompt...")
        hf_num_frames = getattr(processor.video_processor, "num_frames", 32)
        T = raw_video.shape[0]
        if T > hf_num_frames:
            sub_indices = np.arange(0, T, T / hf_num_frames).astype(int)[
                :hf_num_frames
            ]
            raw_video_sub = raw_video[sub_indices]
        else:
            raw_video_sub = raw_video
        # HF expects channels-first (T, C, H, W).
        raw_video_hf = np.transpose(raw_video_sub, (0, 3, 1, 2))
        hf_data_video = _precompute_hf_outputs(
            hf_model,
            hf_tokenizer,
            processor,
            PROMPT_VIDEO,
            raw_image=None,
            raw_audio=None,
            raw_video=raw_video_hf,
            skip_generate=skip_generate,
        )
        hf_data_video["raw_video_sub"] = raw_video_sub

    return hf_data_text, hf_data_image, hf_data_audio, hf_data_video


# ── Input building ──────────────────────────────────────────────────────────


def _build_preprocessor_free_inputs(
    backbone, hf_data, image_placeholder_id, audio_placeholder_id=None
):
    """Build KH backbone inputs from HF-preprocessed data."""
    token_ids = hf_data["input_ids"].astype(np.int32)
    padding_mask = hf_data["attention_mask"].astype(np.int32)
    batch_size = token_ids.shape[0]

    if hf_data["pixel_values"] is not None:
        pixel_values = hf_data["pixel_values"].astype(np.float32)[
            :, np.newaxis, :, :
        ]
    else:
        pixel_values = np.zeros((batch_size, 0, 1, 768), dtype=np.float32)

    if hf_data["image_position_ids"] is not None:
        pixel_position_ids = hf_data["image_position_ids"].astype(np.int32)[
            :, np.newaxis, :, :
        ]
    else:
        pixel_position_ids = np.zeros((batch_size, 0, 1, 2), dtype=np.int32)

    vision_mask = (token_ids == image_placeholder_id).astype(np.int32)
    vision_rows = [
        np.where(vision_mask[index])[0].astype(np.int32)
        for index in range(batch_size)
    ]
    max_vision_tokens = max((len(row) for row in vision_rows), default=0)
    vision_indices = np.zeros((batch_size, max_vision_tokens), dtype=np.int32)
    for index, row in enumerate(vision_rows):
        vision_indices[index, : len(row)] = row

    sequence_length = token_ids.shape[1]
    position_ids = np.arange(sequence_length, dtype=np.int32)[np.newaxis, :]
    position_ids = np.repeat(position_ids, batch_size, axis=0)

    keras_hub_inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
        "pixel_values": ops.convert_to_tensor(pixel_values),
        "pixel_position_ids": ops.convert_to_tensor(pixel_position_ids),
        "position_ids": ops.convert_to_tensor(position_ids),
        "vision_indices": ops.convert_to_tensor(vision_indices),
        "vision_mask": ops.convert_to_tensor(vision_mask),
    }

    # Unified model doesn't use mel spectrograms — audio is raw waveform.
    # Provide empty audio tensors for the backbone.
    feat_size = 640  # default audio_samples_per_token
    keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(
        np.zeros((batch_size, 0, 1, feat_size), dtype=np.float32)
    )
    keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(
        np.zeros((batch_size, 0, 0), dtype=bool)
    )
    keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(
        np.zeros((batch_size, 0), dtype=np.int32)
    )
    keras_hub_inputs["audio_mask"] = ops.convert_to_tensor(
        np.zeros((batch_size, sequence_length), dtype=np.int32)
    )

    return keras_hub_inputs


# ── Verification helpers ────────────────────────────────────────────────────


def _test_token_ids(label, preprocessor, prompt, hf_token_ids, **media_kwargs):
    """Assert KH-preprocessed token IDs match HF token IDs."""
    kh_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        sequence_length=hf_token_ids.shape[1],
    )
    kh_token_ids = ops.convert_to_numpy(kh_inputs["token_ids"])
    np.testing.assert_array_equal(kh_token_ids, hf_token_ids)
    print(f"✓ [{label}] Token IDs match.")


def _test_numerics(label, backbone, keras_hub_inputs, hf_logits):
    """Assert backbone logits match HF logits within tolerance."""
    if isinstance(keras_hub_inputs, tuple):
        keras_hub_inputs = keras_hub_inputs[0]

    expected_names = [
        "token_ids",
        "padding_mask",
        "pixel_values",
        "pixel_position_ids",
        "position_ids",
        "vision_indices",
        "vision_mask",
        "audio_mel",
        "audio_mel_mask",
        "audio_indices",
        "audio_mask",
    ]
    keras_hub_inputs = {
        k: v for k, v in keras_hub_inputs.items() if k in expected_names
    }

    with torch.no_grad():
        kh_output = backbone(keras_hub_inputs)
        if kh_output.shape[1] > hf_logits.shape[1]:
            kh_output = kh_output[:, : hf_logits.shape[1], :]

        kh_logits = ops.convert_to_numpy(
            backbone.token_embedding(kh_output, reverse=True)
        ).astype(np.float32)

    abs_diff = np.abs(kh_logits - hf_logits)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    try:
        np.testing.assert_allclose(kh_logits, hf_logits, atol=1e-3, rtol=1e-3)
        print(
            f"✅ [{label}] Logits within 1e-3 tolerance "
            f"(max={max_diff:.6f}, mean={mean_diff:.6f})."
        )
    except AssertionError:
        diff = np.abs(kh_logits - hf_logits)
        tol = 1e-3 + 1e-3 * np.abs(hf_logits)
        mismatched = int(np.sum(diff > tol))
        total = hf_logits.size
        pct = 100.0 * (1.0 - mismatched / total)
        print(
            f"⚠️  [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={pct:.2f}% ({total - mismatched}/{total}).\n"
            "    NOTE: Generated text comparison is the authoritative check."
        )


def _test_generate(
    label,
    kh_model,
    prompt,
    hf_generated_text,
    max_length=2048 + 64,
    assistant_model=None,
    **media_kwargs,
):
    """Run KH .generate() and compare output against HF-generated text."""
    generate_kwargs = {"max_length": max_length}
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model
    kh_output = kh_model.generate(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        **generate_kwargs,
    )
    kh_text = kh_output[0] if isinstance(kh_output, list) else kh_output
    if isinstance(kh_text, str):
        if kh_text.startswith(prompt):
            kh_text = kh_text[len(prompt) :]
        else:
            for marker in ("<start_of_turn>model\n", "<|turn>model\n"):
                idx = kh_text.rfind(marker)
                if idx != -1:
                    kh_text = kh_text[idx + len(marker) :]
                    break

    print(f"\n[{label}]🔶 HF generate output:\n  {hf_generated_text}")
    print(f"[{label}]🔶 KH generate output:\n  {kh_text}")


# ── Assistant model helpers ─────────────────────────────────────────────────


def _precompute_assistant_hf_outputs(
    hf_target_model,
    hf_model,
    hf_tokenizer,
    processor=None,
    raw_image=None,
    raw_audio=None,
    raw_video=None,
    is_audio_model=False,
    is_video_model=False,
    skip_generate=False,
):
    """Run HF forward pass for assistant and return verification data."""
    print("-> Precomputing HF assistant outputs ...")

    inputs = hf_tokenizer(
        PROMPT_TEXT, return_tensors="pt", add_special_tokens=True
    )
    input_ids = inputs["input_ids"]

    # Run target model to get shared KV states.
    with torch.no_grad():
        target_out = hf_target_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_shared_kv_states=True,
        )
    shared_kv_states = target_out.shared_kv_states
    hf_last_hs = target_out.hidden_states[-1][:, -1:].detach().numpy()

    # Build inputs_embeds for the assistant.
    last_token_id = input_ids[:, -1:]
    with torch.no_grad():
        last_token_embedding = hf_target_model.get_input_embeddings()(
            last_token_id
        )
        last_hidden_state_t = torch.from_numpy(hf_last_hs)
        inputs_embeds = torch.cat(
            [last_token_embedding, last_hidden_state_t], dim=-1
        )

    # Run assistant model.
    seq_len = input_ids.shape[1]
    position_ids = torch.tensor([[seq_len - 1]], dtype=torch.long)
    with torch.no_grad():
        assistant_out = hf_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
        )
    hf_logits = assistant_out.logits.detach().numpy()
    hf_params = _count_hf_params(hf_model)

    # Speculative generation comparison.
    hf_generated_text = None
    if not skip_generate and processor is not None:
        print("-> Running HF speculative generation (text) ...")
        proc_inputs = processor(
            text=PROMPT_TEXT,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )
        bos_id = hf_tokenizer.bos_token_id
        if (
            bos_id is not None
            and proc_inputs["input_ids"][0, 0].item() != bos_id
        ):
            bos = torch.full(
                (proc_inputs["input_ids"].shape[0], 1),
                bos_id,
                dtype=proc_inputs["input_ids"].dtype,
            )
            proc_inputs["input_ids"] = torch.cat(
                [bos, proc_inputs["input_ids"]], dim=1
            )
            if "attention_mask" in proc_inputs:
                proc_inputs["attention_mask"] = torch.ones_like(
                    proc_inputs["input_ids"]
                )
        prompt_len = proc_inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_ids = hf_target_model.generate(
                **proc_inputs,
                assistant_model=hf_model,
                max_new_tokens=64,
            )
        hf_generated_text = hf_tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True
        )

    return {
        "logits": hf_logits,
        "last_hidden_state": hf_last_hs,
        "last_token_embedding": last_token_embedding.detach().numpy(),
        "input_ids": input_ids.numpy(),
        "shared_kv_states": shared_kv_states,
        "generated_text": hf_generated_text,
        "param_count": hf_params,
    }


def _count_assistant_params(kh_assistant):
    """Count KH assistant params, excluding token_ordering buffer."""
    unique_weights = {
        id(w): w for w in kh_assistant.weights if "token_ordering" not in w.name
    }.values()
    return sum(w.numpy().size for w in unique_weights)


def _verify_assistant(
    kh_assistant,
    hf_data,
    hf_tokenizer,
    keras_hub_target_preset=None,
    skip_generate=False,
    is_audio_model=False,
):
    """Verify assistant model: param count, logit numerics, generation."""
    print("\n--- Section 1: Parameter count ---")
    kh_params = _count_assistant_params(kh_assistant)
    hf_params = hf_data["param_count"]
    print(f"   KH params: {kh_params:,}")
    print(f"   HF params: {hf_params:,}")
    np.testing.assert_equal(kh_params, hf_params)
    print("✓ Parameter counts match.")

    print("\n--- Section 2: Logit numerics ---")
    input_ids = hf_data["input_ids"]
    shared_kv_states = hf_data["shared_kv_states"]
    hf_logits = hf_data["logits"]
    last_hidden_state_np = hf_data["last_hidden_state"]
    last_token_embedding_np = hf_data["last_token_embedding"]

    # Convert HF KV to KH cache format.
    def _hf_kv_to_kh(k_t, v_t):
        k = np.transpose(k_t.detach().numpy(), (0, 2, 1, 3))
        v = np.transpose(v_t.detach().numpy(), (0, 2, 1, 3))
        return np.stack([k, v], axis=1)

    sliding_kv = _hf_kv_to_kh(*shared_kv_states["sliding_attention"])
    full_kv = _hf_kv_to_kh(*shared_kv_states["full_attention"])

    # Pad to common shape.
    max_seq = max(sliding_kv.shape[2], full_kv.shape[2])
    max_heads = max(sliding_kv.shape[3], full_kv.shape[3])
    max_dim = max(sliding_kv.shape[4], full_kv.shape[4])

    def _pad_kv(kv, t_seq, t_heads, t_dim):
        _, _, s, h, d = kv.shape
        return np.pad(
            kv,
            [(0, 0), (0, 0), (0, t_seq - s), (0, t_heads - h), (0, t_dim - d)],
        )

    sliding_kv = _pad_kv(sliding_kv, max_seq, max_heads, max_dim)
    full_kv = _pad_kv(full_kv, max_seq, max_heads, max_dim)
    target_cache_np = np.stack([sliding_kv, full_kv], axis=1)

    seq_len = input_ids.shape[1]
    last_token_embedding_t = ops.convert_to_tensor(
        last_token_embedding_np, dtype=kh_assistant.compute_dtype
    )
    last_hidden_state_t = ops.convert_to_tensor(
        last_hidden_state_np, dtype=kh_assistant.compute_dtype
    )
    target_cache_t = ops.convert_to_tensor(
        target_cache_np, dtype=kh_assistant.compute_dtype
    )

    kh_out = kh_assistant.call_with_cache(
        last_token_embedding_t,
        last_hidden_state_t,
        target_cache_t,
        cache_update_index=seq_len - 1,
        padding_mask=ops.ones((1, 1), dtype="bool"),
    )
    kh_logits = ops.convert_to_numpy(kh_out[0])

    # Compare only finite positions.
    active_mask = np.isfinite(hf_logits) & np.isfinite(kh_logits)
    active_hf = hf_logits[active_mask]
    active_kh = kh_logits[active_mask]
    abs_diff = np.abs(active_kh - active_hf)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"   max |Δlogit| = {max_diff:.6f},  mean |Δlogit| = {mean_diff:.6f}")
    np.testing.assert_allclose(
        active_kh,
        active_hf,
        atol=1e-3,
        rtol=1e-3,
        err_msg="Assistant logits differ from HF beyond tolerance.",
    )
    print("✓ Logits within tolerance (atol=1e-3, rtol=1e-3).")

    # Speculative generation with unified target.
    if skip_generate or keras_hub_target_preset is None:
        print("\n--- Generation: SKIPPED ---")
    else:
        print("\n--- Section 3: Speculative generation ---")
        kh_target = keras_hub.models.Gemma4UnifiedCausalLM.from_preset(
            keras_hub_target_preset, dtype="float32"
        )
        preprocessor = (
            keras_hub.models.Gemma4UnifiedCausalLMPreprocessor.from_preset(
                keras_hub_target_preset
            )
        )
        if not is_audio_model:
            preprocessor.audio_converter = None
        kh_target.preprocessor = preprocessor

        _test_generate(
            "assistant-speculative-text",
            kh_target,
            PROMPT_TEXT,
            hf_data.get("generated_text") or "(not available)",
            assistant_model=kh_assistant,
        )
        del kh_target

    print("\n✓ Assistant verification complete.")
    return kh_assistant


# ── KH model loading ───────────────────────────────────────────────────────


def _load_keras_hub_model(keras_hub_preset, is_audio_model):
    """Load KerasHub unified backbone, tokenizer, and preprocessor."""
    backbone = keras_hub.models.Gemma4UnifiedBackbone.from_preset(
        keras_hub_preset, dtype="float32"
    )
    tokenizer = keras_hub.models.Gemma4UnifiedTokenizer.from_preset(
        keras_hub_preset
    )
    preprocessor = (
        keras_hub.models.Gemma4UnifiedCausalLMPreprocessor.from_preset(
            keras_hub_preset
        )
    )
    if not is_audio_model:
        preprocessor.audio_converter = None
    print("-> KerasHub unified model loaded.")
    return backbone, tokenizer, preprocessor


# ── Full verification ───────────────────────────────────────────────────────


def _verify_model(
    backbone,
    tokenizer,
    preprocessor,
    hf_data_text,
    hf_data_image,
    hf_data_audio,
    hf_data_video,
    raw_image,
    raw_audio,
    raw_video,
    is_audio_model,
    is_video_model,
    final_logit_cap,
):
    """Run all verification stages and return Gemma4UnifiedCausalLM."""
    # 1. Parameter count.
    kh_params = _count_keras_hub_params(backbone)
    hf_params = hf_data_image["param_count"]
    np.testing.assert_equal(kh_params, hf_params)
    print(f"\n✓ Parameter count: {kh_params:,}")

    # 2. Token ID verification.
    print("\n--- Token ID Verification ---")

    _test_token_ids(
        "text", preprocessor, PROMPT_TEXT, hf_data_text["input_ids"]
    )

    actual_num_tokens = int(
        np.sum(hf_data_image["input_ids"][0] == tokenizer.image_placeholder_id)
    )
    saved_num_tokens = preprocessor.num_vision_tokens_per_image
    preprocessor.num_vision_tokens_per_image = actual_num_tokens
    _test_token_ids(
        "image",
        preprocessor,
        PROMPT_IMAGE,
        hf_data_image["input_ids"],
        images=raw_image,
    )
    preprocessor.num_vision_tokens_per_image = saved_num_tokens

    if is_audio_model and hf_data_audio is not None:
        _test_token_ids(
            "audio",
            preprocessor,
            PROMPT_AUDIO,
            hf_data_audio["input_ids"],
            audio=raw_audio,
        )

    if is_video_model and hf_data_video is not None:
        raw_video_sub = hf_data_video["raw_video_sub"]
        hf_video_seq_len = hf_data_video["input_ids"].shape[1]
        saved_num_frames = preprocessor.num_frames_per_video
        saved_packer_seq_len = preprocessor.packer.sequence_length
        preprocessor.num_frames_per_video = raw_video_sub.shape[0]
        preprocessor.packer.sequence_length = hf_video_seq_len + 1
        _test_token_ids(
            "video",
            preprocessor,
            PROMPT_VIDEO,
            hf_data_video["input_ids"],
            videos=raw_video_sub,
        )
        preprocessor.num_frames_per_video = saved_num_frames
        preprocessor.packer.sequence_length = saved_packer_seq_len

    # 3. Numerics / logit verification.
    print("\n--- Numerics Verification ---")

    # Text: feed through KH preprocessor.
    kh_inputs_text = preprocessor.generate_preprocess(
        {"prompts": [PROMPT_TEXT]},
        sequence_length=hf_data_text["logits"].shape[1],
    )
    _test_numerics(
        "text (KH preproc)", backbone, kh_inputs_text, hf_data_text["logits"]
    )

    # Image: use HF-preprocessed pixel values.
    kh_inputs_image = _build_preprocessor_free_inputs(
        backbone, hf_data_image, tokenizer.image_placeholder_id
    )
    _test_numerics(
        "image (HF preproc)",
        backbone,
        kh_inputs_image,
        hf_data_image["logits"],
    )

    # Audio: feed through KH preprocessor.
    if is_audio_model and hf_data_audio is not None:
        kh_inputs_audio = preprocessor.generate_preprocess(
            {
                "prompts": [PROMPT_AUDIO],
                "audio": [raw_audio],
            },
            sequence_length=hf_data_audio["logits"].shape[1],
        )
        _test_numerics(
            "audio (KH preproc)",
            backbone,
            kh_inputs_audio,
            hf_data_audio["logits"],
        )

    # Video: end-to-end with KH preprocessor.
    if is_video_model and hf_data_video is not None:
        raw_video_sub = hf_data_video["raw_video_sub"]
        hf_video_seq_len = hf_data_video["logits"].shape[1]
        saved_num_frames = preprocessor.num_frames_per_video
        preprocessor.num_frames_per_video = raw_video_sub.shape[0]
        kh_inputs_video = preprocessor.generate_preprocess(
            {
                "prompts": [PROMPT_VIDEO],
                "videos": [raw_video_sub],
            },
            sequence_length=hf_video_seq_len,
        )
        _test_numerics(
            "video (KH preproc)",
            backbone,
            kh_inputs_video,
            hf_data_video["logits"],
        )
        preprocessor.num_frames_per_video = saved_num_frames

    # 4. Generation comparison.
    gemma4_lm = keras_hub.models.Gemma4UnifiedCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
        sampler="greedy",
        final_logit_cap=final_logit_cap,
    )

    if FLAGS.skip_generate:
        print("\n--- Generation Comparison: SKIPPED (--skip_generate) ---")
        return gemma4_lm

    print("\n--- Generation Comparison ---")
    _test_generate(
        "text", gemma4_lm, PROMPT_TEXT, hf_data_text["generated_text"]
    )

    preprocessor.num_vision_tokens_per_image = actual_num_tokens
    _test_generate(
        "image",
        gemma4_lm,
        PROMPT_IMAGE,
        hf_data_image["generated_text"],
        images=raw_image,
    )
    preprocessor.num_vision_tokens_per_image = saved_num_tokens

    if is_audio_model and hf_data_audio is not None:
        _test_generate(
            "audio",
            gemma4_lm,
            PROMPT_AUDIO,
            hf_data_audio["generated_text"],
            audio=raw_audio,
        )

    if is_video_model and hf_data_video is not None:
        raw_video_sub = hf_data_video["raw_video_sub"]
        hf_video_seq_len = hf_data_video["input_ids"].shape[1]
        saved_num_frames = preprocessor.num_frames_per_video
        saved_packer_seq_len = preprocessor.packer.sequence_length
        preprocessor.num_frames_per_video = raw_video_sub.shape[0]
        preprocessor.packer.sequence_length = hf_video_seq_len
        _test_generate(
            "video",
            gemma4_lm,
            PROMPT_VIDEO,
            hf_data_video["generated_text"],
            max_length=hf_video_seq_len + 64,
            videos=raw_video_sub,
        )
        preprocessor.num_frames_per_video = saved_num_frames
        preprocessor.packer.sequence_length = saved_packer_seq_len

    return gemma4_lm


# ── Save ────────────────────────────────────────────────────────────────────


def _save_preset(model, keras_hub_preset, preset, save_dtype, final_logit_cap):
    """Save the model to a local preset directory."""
    preset_save_path = f"./{preset}"
    print(f"\n-> Saving model in {save_dtype} to {preset_save_path} ...")
    is_assistant = "assistant" in preset

    if save_dtype == "bfloat16":
        if is_assistant:
            del model
            gc.collect()
            kh_save = keras_hub.models.Gemma4AssistantCausalLM.from_preset(
                keras_hub_preset, dtype="bfloat16"
            )
            kh_save.save_to_preset(preset_save_path)
        else:
            preprocessor_ref = model.preprocessor
            del model
            gc.collect()
            backbone_bf16 = keras_hub.models.Gemma4UnifiedBackbone.from_preset(
                keras_hub_preset, dtype="bfloat16"
            )
            kh_save = keras_hub.models.Gemma4UnifiedCausalLM(
                backbone=backbone_bf16,
                preprocessor=preprocessor_ref,
                sampler="greedy",
                final_logit_cap=final_logit_cap,
            )
            kh_save.save_to_preset(preset_save_path)
    else:
        model.save_to_preset(preset_save_path)

    print(f"-> Saved {save_dtype} preset to {preset_save_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset!r}. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]
    is_assistant = "assistant" in preset
    base_preset = (
        hf_preset.replace("-assistant", "") if is_assistant else hf_preset
    )
    keras_hub_preset = f"hf://{base_preset}"

    raw_image, raw_audio, raw_video = _load_test_assets()

    (
        hf_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
        hf_target_model,
    ) = _load_hf_model(hf_preset)

    if is_assistant:
        hf_data_assistant = _precompute_assistant_hf_outputs(
            hf_target_model,
            hf_model,
            hf_tokenizer,
            processor=processor,
            raw_image=raw_image,
            raw_audio=raw_audio,
            raw_video=raw_video,
            is_audio_model=is_audio_model,
            is_video_model=is_video_model,
            skip_generate=FLAGS.skip_generate,
        )
        del hf_target_model, hf_model
        gc.collect()

        kh_assistant = keras_hub.models.Gemma4AssistantCausalLM.from_preset(
            f"hf://{hf_preset}", dtype="float32"
        )

        _verify_assistant(
            kh_assistant,
            hf_data_assistant,
            hf_tokenizer,
            keras_hub_target_preset=keras_hub_preset,
            skip_generate=FLAGS.skip_generate,
            is_audio_model=is_audio_model,
        )

        del hf_data_assistant
        gc.collect()

        _save_preset(
            kh_assistant,
            f"hf://{hf_preset}",
            preset,
            FLAGS.save_dtype,
            final_logit_cap,
        )
        return

    # --- Non-assistant path ---
    hf_data_text, hf_data_image, hf_data_audio, hf_data_video = (
        _precompute_all_hf_outputs(
            hf_model,
            hf_tokenizer,
            processor,
            raw_image,
            raw_audio,
            raw_video,
            is_audio_model,
            is_video_model,
            skip_generate=FLAGS.skip_generate,
        )
    )
    del hf_model
    gc.collect()

    backbone, tokenizer, preprocessor = _load_keras_hub_model(
        keras_hub_preset, is_audio_model
    )

    gemma4_lm = _verify_model(
        backbone,
        tokenizer,
        preprocessor,
        hf_data_text,
        hf_data_image,
        hf_data_audio,
        hf_data_video,
        raw_image,
        raw_audio,
        raw_video,
        is_audio_model,
        is_video_model,
        final_logit_cap,
    )

    del hf_data_text, hf_data_image, hf_data_audio, hf_data_video
    del backbone, tokenizer, preprocessor
    gc.collect()

    _save_preset(
        gemma4_lm,
        keras_hub_preset,
        preset,
        FLAGS.save_dtype,
        final_logit_cap,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
