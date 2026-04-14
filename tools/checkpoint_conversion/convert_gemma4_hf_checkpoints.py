"""Convert Gemma4 HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_gemma4_hf_checkpoints.py \
        --preset gemma4_instruct_2b \
        --save_dtype bfloat16
"""

import contextlib
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
    "gemma4_2b": "google/gemma-4-E2B",
    "gemma4_instruct_2b": "google/gemma-4-E2B-it",
    "gemma4_4b": "google/gemma-4-E4B",
    "gemma4_instruct_4b": "google/gemma-4-E4B-it",
    "gemma4_26b_a4b": "google/gemma-4-26B-A4B",
    "gemma4_instruct_26b_a4b": "google/gemma-4-26B-A4B-it",
    "gemma4_31b": "google/gemma-4-31B",
    "gemma4_instruct_31b": "google/gemma-4-31B-it",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# 10-second, 1 MB clip used for video verification when --video_path is not set.
VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
AUDIO_FILE_PATH = os.path.join(
    _REPO_ROOT,
    "keras_hub/src/tests/test_data/audio_transcription_tests/male_short_voice_clip_3sec.wav",
)

# Text-only prompt (no media placeholder) for KH-preprocessed token-id check.
PROMPT_TEXT = (
    "<start_of_turn>user\nWhat is the capital of France?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

# Image prompt: use HF-preprocessed inputs for numerics (HF uses PIL resize,
# KH uses its own image converter, so pixel values differ slightly).
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
    "Skip the generation comparison step. Useful for large models where "
    "generation is slow or unnecessary (numerics verification is sufficient).",
)


def _evict_hf_cache(repo_id):
    """Delete all cached revisions for `repo_id` from the HF hub cache.

    This ensures that the subsequent `from_preset("hf://...")` call fetches
    the same weights that AutoModel loaded above (force_download=True), rather
    than reading a potentially stale local copy.
    """
    import huggingface_hub

    try:
        cache_info = huggingface_hub.scan_cache_dir()
    except Exception:
        return  # Non-fatal; skip if cache scan fails.

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            commit_hashes = [rev.commit_hash for rev in repo.revisions]
            if commit_hashes:
                strategy = cache_info.delete_revisions(*commit_hashes)
                strategy.execute()
                print(
                    f"-> Evicted {len(commit_hashes)} cached revision(s) "
                    f"for {repo_id} from HF hub cache."
                )
            break


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _download_test_video():
    """Download the Big Buck Bunny test clip and decode to (T, H, W, C) uint8.

    Requires PyAV (`pip install av`).  Returns None on any failure so that
    video verification is simply skipped rather than crashing the script.
    """
    try:
        import av
    except ImportError:
        print(
            "Warning: PyAV not installed; skipping video download. "
            "Install with `pip install av`."
        )
        return None
    try:
        response = requests.get(VIDEO_URL, timeout=60)
        response.raise_for_status()
        container = av.open(BytesIO(response.content))
        frames = [
            f.to_ndarray(format="rgb24") for f in container.decode(video=0)
        ]
        return np.stack(frames)  # (T, H, W, C), channels-last for KH
    except Exception as e:
        print(
            f"Warning: could not download test video ({e}); "
            f"skipping video verification."
        )
        return None


def _count_hf_params(hf_model):
    param_names = {name for name, _ in hf_model.named_parameters()}
    num_params = sum(param.numel() for param in hf_model.parameters())
    num_buffers = sum(
        value.numel()
        for name, value in hf_model.state_dict().items()
        if name not in param_names
        and (
            name.endswith(".layer_scalar")
            or (
                (
                    "vision_tower.encoder.layers" in name
                    or "audio_tower.layers" in name
                )
                and name.endswith(
                    (".input_min", ".input_max", ".output_min", ".output_max")
                )
            )
            # std_bias / std_scale are registered buffers on the vision tower
            # for 26B-A4B and 31B models (standardize=True).
            or name
            in (
                "model.vision_tower.std_bias",
                "model.vision_tower.std_scale",
            )
        )
    )
    return num_params + num_buffers


def _count_keras_hub_params(backbone):
    unique_weights = {
        id(weight): weight for weight in backbone.weights
    }.values()
    return sum(weight.numpy().size for weight in unique_weights)


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

    # Ensure HF inputs start with BOS to match KerasHub behavior.
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

    # Register a forward hook on embed_vision to capture the per-frame
    # embeddings at text hidden size.  These are used in the
    # "video (HF encoder injected)" numeric test to verify that the KH decoder
    # is numerically correct independent of any video-preprocessing differences.
    _hooks = []
    _hf_model_inner = getattr(hf_model, "model", None)
    if _hf_model_inner is not None and raw_video is not None:
        if hasattr(_hf_model_inner, "embed_vision"):
            _video_frame_embeds = []

            def _video_hook(mod, inp, out):
                _video_frame_embeds.append(out.detach().cpu().float().numpy())

            _hooks.append(
                _hf_model_inner.embed_vision.register_forward_hook(_video_hook)
            )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=False)

    for h in _hooks:
        h.remove()
    _hooks.clear()

    hf_logits = hf_outputs.logits.detach().cpu().float().numpy()
    hf_input_ids = hf_inputs["input_ids"].detach().cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].detach().cpu().numpy()

    # Raw audio mel features are retained for the mel-spectrogram comparison
    # test.
    hf_audio_features = None
    if "input_features" in hf_inputs:
        with torch.no_grad():
            if hasattr(hf_model, "model") and hasattr(
                hf_model.model, "audio_tower"
            ):
                hf_af = hf_model.model.audio_tower(hf_inputs["input_features"])
                if hasattr(hf_af, "last_hidden_state"):
                    hf_af = hf_af.last_hidden_state
                hf_audio_features = hf_af.detach().cpu().float().numpy()

    # Capture video embeddings (after embed_vision projection, at text hidden
    # dim).
    _vfe = locals().get("_video_frame_embeds")
    if raw_video is not None and _vfe:
        # embed_vision may be called once for the full batch of frames
        # (returning (N, T, H) or (N*T, H)) or once per frame (returning (T,
        # H) each time).
        # Concatenate all captured outputs along axis-0 then force a 3-D shape
        # (1, total_tokens, H) regardless of the original layout.
        stacked = np.concatenate(_vfe, axis=0)  # (N*T, H) or (N, T, H)
        if stacked.ndim == 2:
            total_tokens, Hd = stacked.shape
        else:
            # 3-D: flatten frame/patch dims together
            stacked = stacked.reshape(-1, stacked.shape[-1])
            total_tokens, Hd = stacked.shape
        hf_video_embeddings = stacked.reshape(
            1, total_tokens, Hd
        )  # (1, N*T, H)
    else:
        hf_video_embeddings = None

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
                do_sample=False,
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
        "hf_audio_features": hf_audio_features,
        "hf_video_embeddings": hf_video_embeddings,
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
    # For video: record how many frames HF actually sampled.
    # Video processor uses "pixel_values_videos" (not "pixel_values").
    # Its shape is (num_videos, num_frames, max_patches, patch_pixels) → 4D,
    # so shape[1] is always the number of sampled frames.
    if raw_video is not None and "pixel_values_videos" in hf_inputs:
        pv = hf_inputs["pixel_values_videos"]
        ret["num_video_frames"] = int(pv.shape[1])
    return ret


def _build_preprocessor_free_inputs(
    backbone, hf_data, image_placeholder_id, audio_placeholder_id=None
):
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

    if "input_features" in hf_data:
        audio_mel = hf_data["input_features"][:, np.newaxis, :, :]
        keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(audio_mel)

        # All frames are valid (HF pads to fixed length).
        audio_mel_mask = np.ones(
            (batch_size, 1, audio_mel.shape[2]), dtype=bool
        )
        keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(
            audio_mel_mask
        )

        if audio_placeholder_id is not None:
            audio_mask = (token_ids == audio_placeholder_id).astype(np.int32)
            audio_rows = [
                np.where(audio_mask[index])[0].astype(np.int32)
                for index in range(batch_size)
            ]
            max_audio_tokens = max((len(row) for row in audio_rows), default=0)
            audio_indices = np.zeros(
                (batch_size, max_audio_tokens), dtype=np.int32
            )
            for index, row in enumerate(audio_rows):
                audio_indices[index, : len(row)] = row
            keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(
                audio_indices
            )
            keras_hub_inputs["audio_mask"] = ops.convert_to_tensor(audio_mask)
    else:
        feat_size = getattr(backbone.audio_encoder, "input_feat_size", 128)
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


@contextlib.contextmanager
def _mock_encoder_call(encoder, hf_embeddings, n_clips=1):
    """Temporarily replace encoder.call so the backbone uses pre-computed HF
    embeddings instead of running the KH encoder.

    This lets us verify the KH decoder in isolation: if the test passes when
    HF's encoder output is injected but fails with KH's preprocessor output,
    the divergence is in preprocessing (e.g. pixel normalisation), not the
    decoder weights.

    Args:
        encoder: KH encoder layer whose `.call` will be monkey-patched.
        hf_embeddings: numpy array (B, T, H) — HF encoder output at text
            hidden size, as captured by a forward hook on `embed_vision`.
        n_clips: number of clips/frames axis expected by the backbone
            (backbone shape: (B, n_clips, T//n_clips, H)).
    """
    B, T, Hd = hf_embeddings.shape
    hf_4d = hf_embeddings.reshape(B, n_clips, T // n_clips, Hd).astype(
        np.float32
    )
    hf_t = ops.convert_to_tensor(hf_4d)
    original_call = encoder.call

    def mock_call(*args, **kwargs):
        return ops.cast(hf_t, encoder.compute_dtype)

    encoder.call = mock_call
    try:
        yield
    finally:
        encoder.call = original_call


def _test_token_ids(label, preprocessor, prompt, hf_token_ids, **media_kwargs):
    """Assert KH-preprocessed token IDs match HF token IDs for any modality."""
    kh_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        sequence_length=hf_token_ids.shape[1],
    )
    kh_token_ids = ops.convert_to_numpy(kh_inputs["token_ids"])
    np.testing.assert_array_equal(kh_token_ids, hf_token_ids)
    print(f"✓ [{label}] Token IDs match.")


def _test_audio_preprocessor(preprocessor, raw_audio, hf_input_features):
    """Assert KH audio mel features match HF within 1e-3 tolerance."""
    kh_mel = ops.convert_to_numpy(preprocessor.audio_converter(raw_audio))
    hf_mel = hf_input_features[0]  # HF shape: (1, frames, mels)
    kh_mel = kh_mel[0] if kh_mel.ndim > 2 else kh_mel

    min_len = min(hf_mel.shape[0], kh_mel.shape[0])
    np.testing.assert_allclose(
        kh_mel[:min_len], hf_mel[:min_len], atol=1e-3, rtol=1e-3
    )
    print("✓ [Audio] Mel features within 1e-3 tolerance.")


def _test_numerics(label, backbone, keras_hub_inputs, hf_logits):
    """Assert backbone logits match HF logits within 1e-3 tolerance.

    `backbone.token_embedding(..., reverse=True)` already applies
    `final_logit_soft_cap` (via `logit_soft_cap` on `ReversibleEmbedding`), so
    the KH logits are already in the same softcapped space as HF's logits.
    No extra transformation is needed here.
    """
    if isinstance(keras_hub_inputs, tuple):
        keras_hub_inputs = keras_hub_inputs[0]

    # Keep only the keys the backbone actually accepts.
    expected_names = [
        "token_ids",
        "padding_mask",
        "pixel_values",
        "pixel_position_ids",
        "position_ids",
        "vision_indices",
        "vision_mask",
    ]
    if getattr(backbone, "audio_encoder", None) is not None:
        expected_names += [
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
        # Trim if KH sequence is longer than HF (e.g. due to padding).
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
    **media_kwargs,
):
    """Run KH .generate() and compare output against HF-generated text.

    `max_length` is the total sequence length cap (prompt + response).  For
    modalities with very long prompts (e.g. video with many frames) the caller
    should pass a larger value so that generation isn't cut off before any
    response tokens are produced.
    """
    kh_output = kh_model.generate(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        max_length=max_length,
    )
    kh_text = kh_output[0] if isinstance(kh_output, list) else kh_output
    if isinstance(kh_text, str):
        # Strip the prompt prefix. For modalities like audio/video the
        # preprocessor expands placeholder tokens, so the output no longer
        # starts with the literal prompt string. Instead, find the last
        # model-turn marker and strip everything up to and including it.
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
        raw_video = np.stack(frames)  # (T, H, W, C) channels-last for KH
    else:
        # No local path supplied — download the standard Big Buck Bunny clip.
        raw_video = _download_test_video()

    return raw_image, raw_audio, raw_video


def _load_hf_model(hf_preset):
    """Load HF model/tokenizer/processor and detect modality capabilities."""
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
            hf_model.config.get_text_config(), "final_logit_softcapping", None
        )
    print(f"-> final_logit_softcapping: {final_logit_cap}")

    return (
        hf_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
    )


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
    """Run HF forward passes for all applicable modalities and return
    results."""
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
        # Pre-subsample to processor's expected frame count so that HF stores
        # frames_indices = [0, 1, ..., N-1].  When KH also receives these same
        # N frames it uses sequential indices for timestamps, so both pipelines
        # produce identical timestamp tokens (i / fps for i in 0..N-1).
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
        # Store subsampled frames (channels-last) for KH verification.
        hf_data_video["raw_video_sub"] = raw_video_sub

    return hf_data_text, hf_data_image, hf_data_audio, hf_data_video


def _load_keras_hub_model(keras_hub_preset, is_audio_model):
    """Load KerasHub backbone, tokenizer, and preprocessor from a preset."""
    backbone = keras_hub.models.Gemma4Backbone.from_preset(
        keras_hub_preset, dtype="float32"
    )
    tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(keras_hub_preset)
    preprocessor = keras_hub.models.Gemma4CausalLMPreprocessor.from_preset(
        keras_hub_preset
    )
    # The preset loader sets audio_converter only when the HF config declares
    # an audio_config; clear it explicitly for text/image-only models.
    if not is_audio_model:
        preprocessor.audio_converter = None
    print("-> KerasHub model loaded.")
    return backbone, tokenizer, preprocessor


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
    """Run all four verification stages and return the assembled
    Gemma4CausalLM."""
    # ── 1. Parameter count ────────────────────────────────────────────────────
    kh_params = _count_keras_hub_params(backbone)
    hf_params = hf_data_image["param_count"]
    np.testing.assert_equal(kh_params, hf_params)
    print(f"\n✓ Parameter count: {kh_params:,}")

    # ── 2. Token ID verification (all modalities) ─────────────────────────────
    print("\n--- Token ID Verification ---")

    _test_token_ids(
        "text", preprocessor, PROMPT_TEXT, hf_data_text["input_ids"]
    )

    # Patch num_vision_tokens_per_image to the actual value used by HF so that
    # KH produces the same number of soft image tokens.
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
        # Use the same pre-subsampled frames that were fed to HF.  Both
        # pipelines then compute timestamps from sequential indices [0..N-1]
        # so timestamp tokens align.  KH derives tokens-per-frame dynamically
        # via _compute_video_n_tokens(), so no num_vision_tokens_per_image
        # patching is needed here.
        raw_video_sub = hf_data_video["raw_video_sub"]
        hf_video_seq_len = hf_data_video["input_ids"].shape[1]
        saved_num_frames_per_video = preprocessor.num_frames_per_video
        # The packer trims at self.sequence_length before the per-call override
        # is applied, so we must widen it to accommodate the full video
        # sequence.
        # Use +1 to avoid dropping the final trailing \n token.
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
        preprocessor.num_frames_per_video = saved_num_frames_per_video
        preprocessor.packer.sequence_length = saved_packer_seq_len

    # ── 3. Numerics / logit verification ──────────────────────────────────────
    print("\n--- Numerics Verification ---")

    # Text: token IDs match HF so feed through KH preprocessor directly.
    kh_inputs_text = preprocessor.generate_preprocess(
        {"prompts": [PROMPT_TEXT]},
        sequence_length=hf_data_text["logits"].shape[1],
    )
    _test_numerics(
        "text (KH preproc)", backbone, kh_inputs_text, hf_data_text["logits"]
    )

    # Image: use HF-preprocessed pixel values to avoid PIL vs KH resize delta.
    kh_inputs_image = _build_preprocessor_free_inputs(
        backbone, hf_data_image, tokenizer.image_placeholder_id
    )
    _test_numerics(
        "image (HF preproc)", backbone, kh_inputs_image, hf_data_image["logits"]
    )

    # Audio: KH mel pipeline aligns with HF within 1e-3.
    if is_audio_model and hf_data_audio is not None:
        _test_audio_preprocessor(
            preprocessor, raw_audio, hf_data_audio["input_features"]
        )
        kh_inputs_audio = preprocessor(
            {
                "prompts": [PROMPT_AUDIO],
                "audio": [raw_audio],
                "responses": [""],
            },
            sequence_length=hf_data_audio["logits"].shape[1] + 1,
        )
        _test_numerics(
            "audio (KH preproc)",
            backbone,
            kh_inputs_audio,
            hf_data_audio["logits"],
        )

    # Video: two numeric checks are run.
    #   1. "video (KH preproc)" — end-to-end using KH's video preprocessor.
    #      A logit mismatch here is expected and acceptable: KH and HF use
    #      slightly different frame resizing/normalisation pipelines, so the
    #      pixel values fed to the vision encoder differ.
    #   2. "video (HF encoder injected)" — the backbone is run with HF's own
    #      vision-encoder output injected in place of KH's.  This test is the
    #      authoritative decoder correctness check: it passes iff the KH
    #      decoder weights and architecture are correct, independent of
    #      preprocessing differences.
    if is_video_model and hf_data_video is not None:
        raw_video_sub = hf_data_video["raw_video_sub"]
        hf_video_seq_len = hf_data_video["logits"].shape[1]
        saved_num_frames_per_video = preprocessor.num_frames_per_video
        saved_packer_seq_len = preprocessor.packer.sequence_length
        preprocessor.num_frames_per_video = raw_video_sub.shape[0]
        preprocessor.packer.sequence_length = hf_video_seq_len + 1
        kh_inputs_video = preprocessor(
            {
                "prompts": [PROMPT_VIDEO],
                "videos": [raw_video_sub],
                "responses": [""],
            },
            sequence_length=hf_video_seq_len + 1,
        )
        # Test 1: end-to-end with KH preprocessor.
        _test_numerics(
            "video (KH preproc)",
            backbone,
            kh_inputs_video,
            hf_data_video["logits"],
        )
        # Test 2: inject HF vision-encoder outputs to verify decoder
        # correctness.
        # n_clips = number of frames (each frame is processed as a separate
        # image by the vision encoder, giving shape (B, n_clips, T_per_frame,
        # H)).
        if hf_data_video.get("hf_video_embeddings") is not None:
            n_frames = raw_video_sub.shape[0]
            with _mock_encoder_call(
                backbone.vision_encoder,
                hf_data_video["hf_video_embeddings"],
                n_clips=n_frames,
            ):
                _test_numerics(
                    "video (HF encoder injected)",
                    backbone,
                    kh_inputs_video,
                    hf_data_video["logits"],
                )
        preprocessor.num_frames_per_video = saved_num_frames_per_video
        preprocessor.packer.sequence_length = saved_packer_seq_len

    # ── 4. Generation comparison (all modalities) ─────────────────────────────
    gemma4_lm = keras_hub.models.Gemma4CausalLM(
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
        saved_num_frames_per_video = preprocessor.num_frames_per_video
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
        preprocessor.num_frames_per_video = saved_num_frames_per_video
        preprocessor.packer.sequence_length = saved_packer_seq_len

    return gemma4_lm


def _save_preset(
    gemma4_lm, keras_hub_preset, preset, save_dtype, final_logit_cap
):
    """Save the model to a local preset directory in the requested dtype."""
    preset_save_path = f"./{preset}"
    print(f"\n-> Saving model in {save_dtype} to {preset_save_path} ...")

    if save_dtype == "bfloat16":
        preprocessor_ref = gemma4_lm.preprocessor
        del gemma4_lm
        gc.collect()
        backbone_bf16 = keras_hub.models.Gemma4Backbone.from_preset(
            keras_hub_preset, dtype="bfloat16"
        )
        gemma4_lm_bf16 = keras_hub.models.Gemma4CausalLM(
            backbone=backbone_bf16,
            preprocessor=preprocessor_ref,
            sampler="greedy",
            final_logit_cap=final_logit_cap,
        )
        gemma4_lm_bf16.save_to_preset(preset_save_path)
    else:
        gemma4_lm.save_to_preset(preset_save_path)

    print(f"-> Saved {save_dtype} preset to {preset_save_path}")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset!r}. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]
    keras_hub_preset = f"hf://{hf_preset}"

    raw_image, raw_audio, raw_video = _load_test_assets()

    (
        hf_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
    ) = _load_hf_model(hf_preset)
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
    gc.collect()

    _save_preset(
        gemma4_lm, keras_hub_preset, preset, FLAGS.save_dtype, final_logit_cap
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
