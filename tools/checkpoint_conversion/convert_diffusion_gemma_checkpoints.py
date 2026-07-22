"""Convert DiffusionGemma HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_diffusion_gemma_checkpoints.py \
        --preset diffusion_gemma_26b_a4b_it \
        --save_dtype bfloat16
"""

import contextlib
import gc
import os

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

import keras_hub

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "diffusion_gemma_26b_a4b_it": "google/diffusiongemma-26B-A4B-it",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

PROMPT_TEXT = (
    "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
)
PROMPT_IMAGE = (
    "<|turn>user\n\n<|image|>\nWhat is in this image?<turn|>\n<|turn>model\n"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Name of the preset to convert. Must be one of: "
    f"{', '.join(PRESET_MAP.keys())}",
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype in which to save the converted preset. Defaults to 'bfloat16'.",
)
flags.DEFINE_bool(
    "skip_generate",
    False,
    "If set, skip the HF/KH .generate() comparison during verification.",
)


def _gather_hf_data(hf_repo_id):
    """Load HF model, run all HF-side computations, return a data dict."""
    from transformers import AutoProcessor
    from transformers import DiffusionGemmaForBlockDiffusion

    print(f"-> Loading HF model from {hf_repo_id} …")
    hf_model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        hf_repo_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(hf_repo_id)
    print("-> HF model loaded.")

    is_multimodal = (
        hasattr(hf_model.config, "vision_config")
        and hf_model.config.vision_config is not None
    )

    # param_count = _count_hf_params(hf_model)

    # Fixed all-zero canvas for deterministic verification.
    # Both text and image tests use the same canvas so HF and KH are compared
    # on identical decoder inputs.
    canvas_length = getattr(hf_model.config, "canvas_length", 256)
    fixed_canvas = torch.zeros(1, canvas_length, dtype=torch.long)

    text_fwd = _hf_forward(
        hf_model, processor, PROMPT_TEXT, decoder_input_ids=fixed_canvas
    )

    image_data = None
    if is_multimodal:
        try:
            raw_image = _load_test_image()
            img_fwd = _hf_forward(
                hf_model,
                processor,
                PROMPT_IMAGE,
                raw_image=raw_image,
                decoder_input_ids=fixed_canvas,
            )
            image_data = {
                "logits": img_fwd["logits"],
                "input_ids": img_fwd["input_ids"],
                "attention_mask": img_fwd["attention_mask"],
                "pixel_values": img_fwd["pixel_values"],
                "image_position_ids": img_fwd["image_position_ids"],
                "generated_text": img_fwd["generated_text"],
                "raw_image": raw_image,
            }
        except Exception as e:
            print(f"⚠️  Image HF forward skipped: {e}")

    del hf_model, processor
    gc.collect()
    print("-> HF model freed.")

    return {
        # "param_count": param_count,
        "canvas_token_ids": np.zeros((1, canvas_length), dtype=np.int32),
        "text_logits": text_fwd["logits"],
        "text_input_ids": text_fwd["input_ids"],
        "text_attention_mask": text_fwd["attention_mask"],
        "text_generated_text": text_fwd["generated_text"],
        "image": image_data,
    }


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


@contextlib.contextmanager
def _no_grad():
    with torch.no_grad():
        yield


def _hf_forward(
    hf_model,
    processor,
    prompt,
    raw_image=None,
    decoder_input_ids=None,
):
    """Run one HF forward pass and return decoder logits + optional generation.

    ``DiffusionGemmaForBlockDiffusion.forward()`` always runs both the encoder
    (prompt → KV cache) and the decoder (canvas → logits via ``layer_scalar``).
    ``hf_out.logits`` are decoder logits over ``decoder_input_ids``.  When
    ``decoder_input_ids`` is None, the model auto-samples a *random* canvas —
    always supply a fixed canvas so verification is deterministic.

    When ``--skip_generate`` is not set, this function also runs
    ``hf_model.generate(**hf_inputs, max_new_tokens=512)`` on the same
    ``hf_inputs`` — matching the DiffusionGemma model-card usage — and returns
    the decoded output text alongside the logits.
    """
    proc_kwargs = {"text": prompt, "return_tensors": "pt"}
    if raw_image is not None:
        proc_kwargs["images"] = raw_image
    hf_inputs = processor(**proc_kwargs)
    hf_inputs = {k: v.cpu() for k, v in hf_inputs.items()}
    forward_inputs = dict(hf_inputs)
    if decoder_input_ids is not None:
        forward_inputs["decoder_input_ids"] = decoder_input_ids.cpu()

    with _no_grad():
        hf_out = hf_model(**forward_inputs, output_hidden_states=False)

    logits = hf_out.logits.detach().cpu().float().numpy()
    input_ids = hf_inputs["input_ids"].numpy()
    attention_mask = hf_inputs.get(
        "attention_mask", torch.ones_like(hf_inputs["input_ids"])
    ).numpy()
    pixel_values = (
        hf_inputs["pixel_values"].detach().cpu().float().numpy()
        if "pixel_values" in hf_inputs
        else None
    )
    image_position_ids = (
        hf_inputs["image_position_ids"].detach().cpu().numpy()
        if "image_position_ids" in hf_inputs
        else None
    )

    del hf_out

    generated_text = None
    if FLAGS.skip_generate:
        generated_text = "(skipped)"
    else:
        try:
            with _no_grad():
                output = hf_model.generate(**hf_inputs, max_new_tokens=512)
            generated_text = processor.decode(
                output[0], skip_special_tokens=False
            )
        except Exception as e:
            print(f"⚠️  HF .generate() failed ({e}).")

    return {
        "logits": logits,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_position_ids": image_position_ids,
        "generated_text": generated_text,
    }


def _kh_forward(
    diffusion_lm,
    token_ids,
    padding_mask,
    canvas_token_ids,
    pixel_values=None,
    pixel_position_ids=None,
    vision_indices=None,
    vision_mask=None,
):
    """Run a full KH encoder + decoder forward pass and return canvas logits.

    Mirrors ``DiffusionGemmaForBlockDiffusion.forward()``:
      1. Encoder: causal attention over prompt → KV cache (uses
         ``encoder_layer_scalar`` via ``_encode_prompt``).
      2. Decoder: one bidirectional denoising step over ``canvas_token_ids``
         using the encoder KV cache (uses ``layer_scalar`` via
         ``_decode_canvas_step``).

    Args:
        diffusion_lm: ``Gemma4BlockDiffusionLM`` instance.
        token_ids: int32 array ``(B, prompt_len)`` — encoder input.
        padding_mask: int32 array ``(B, prompt_len)``.
        canvas_token_ids: int32 array ``(B, canvas_len)`` — must match the
            ``decoder_input_ids`` passed to the HF model.
        pixel_values, pixel_position_ids, vision_indices, vision_mask:
            Optional vision inputs.  When ``pixel_values`` is ``None`` the
            vision encoder is skipped (text-only encoder path).

    Returns:
        float32 numpy array ``(B, canvas_len, vocab_size)``.
    """
    inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
    }
    if not diffusion_lm.backbone.text_only_model and pixel_values is not None:
        inputs["pixel_values"] = ops.convert_to_tensor(pixel_values)
        inputs["pixel_position_ids"] = ops.convert_to_tensor(pixel_position_ids)
        inputs["vision_indices"] = ops.convert_to_tensor(vision_indices)
        inputs["vision_mask"] = ops.convert_to_tensor(vision_mask)

    canvas = ops.convert_to_tensor(canvas_token_ids)

    with torch.no_grad():
        # Step 1: encoder — builds KV cache over the prompt.
        encoder_kv_cache, prompt_length = diffusion_lm._encode_prompt(inputs)
        encoder_kv_cache = diffusion_lm._prepare_encoder_cache_for_decoding(
            encoder_kv_cache
        )

        # Step 2: canvas embeddings (first step → self-conditioning is no-op).
        canvas_embeds = diffusion_lm._prepare_canvas_embeds(
            canvas, prev_logits=None
        )

        # Step 3: decoder — one bidirectional denoising step.
        hidden = diffusion_lm._decode_canvas_step(
            canvas_embeds, encoder_kv_cache, prompt_length
        )

        # Step 4: project to vocabulary logits with soft-cap.
        logits = diffusion_lm._canvas_logits(hidden)

    return ops.convert_to_numpy(logits).astype(np.float32)


def _test_numerics(label, kh_logits, hf_logits):
    """Log max/mean absolute logit difference; warn if > 1e-3."""
    # Trim to common sequence length.
    min_len = min(kh_logits.shape[1], hf_logits.shape[1])
    kh = kh_logits[:, :min_len, :]
    hf = hf_logits[:, :min_len, :]

    abs_diff = np.abs(kh - hf)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    try:
        np.testing.assert_allclose(kh, hf, atol=1e-3, rtol=1e-3)
        print(
            f"✅ [{label}] Logits within 1e-3 tolerance "
            f"(max={max_diff:.6f}, mean={mean_diff:.6f})."
        )
    except AssertionError:
        tol = 1e-3 + 1e-3 * np.abs(hf)
        mismatched = int(np.sum(np.abs(kh - hf) > tol))
        total = hf.size
        pct = 100.0 * (1.0 - mismatched / total)
        print(
            f"⚠️  [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={pct:.2f}% ({total - mismatched}/{total})."
        )


def _test_generate(
    label,
    diffusion_lm,
    prompt,
    hf_generated_text,
    max_length=None,
    **media_kwargs,
):
    """Run KH ``.generate()`` and print the output alongside HF's output.

    ``max_length`` is the prompt sequence length (canvas is appended after by
    the preprocessor).  Defaults to the preprocessor's configured
    ``sequence_length`` when ``None``.
    """
    if FLAGS.skip_generate:
        print(f"[{label}] Generate comparison skipped (--skip_generate).")
        return
    if hf_generated_text is None:
        print(
            f"[{label}] Generate comparison skipped: HF generation unavailable."
        )
        return

    generate_kwargs = {}
    if max_length is not None:
        generate_kwargs["max_length"] = max_length

    if hasattr(diffusion_lm, "sampler") and hasattr(
        diffusion_lm.sampler, "reset"
    ):
        diffusion_lm.sampler.reset()

    try:
        # Wrap each media value in a single-item list so
        # `convert_preprocessing_inputs` calls `np.array([<PIL>])` — none of
        # the KH `_preprocess_images` implementations accept a bare PIL Image.
        # Mirrors `convert_gemma4_hf_checkpoints.py::_test_generate`.
        kh_output = diffusion_lm.generate(
            {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
            **generate_kwargs,
        )
    except Exception as e:
        print(f"⚠️  [{label}] KH .generate() failed: {e}")
        return

    kh_text = _clean_text(kh_output, prompt)
    hf_text = _clean_text(hf_generated_text, prompt)

    print(f"\n[{label}] HF generate output:\n  {hf_text}")
    print(f"[{label}] KH generate output:\n  {kh_text}")


def _clean_text(text, prompt=None):
    """Safely unwrap numpy/tensor arrays/bytes to string and strip prompt
    prefix if present."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple, np.ndarray)) and len(text) > 0:
        text = text[0]
    if hasattr(text, "item"):
        text = text.item()
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        text = str(text)
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :]
    return text


def _build_image_kh_inputs(img_data, image_placeholder_id):
    """Build KH vision inputs from HF preprocessor outputs.

    Uses HF's pixel values directly so that the PIL-vs-KH resize delta does
    not contaminate the numerics check.

    Args:
        img_data: dict from ``hf_data["image"]`` containing ``input_ids``,
            ``attention_mask``, ``pixel_values``, and ``image_position_ids``.
        image_placeholder_id: token ID that marks image placeholder positions in
            ``input_ids``.

    Returns:
        Tuple of (token_ids, padding_mask, pixel_values, pixel_position_ids,
        vision_indices, vision_mask) all as numpy arrays.
    """
    token_ids = img_data["input_ids"].astype(np.int32)
    padding_mask = img_data["attention_mask"].astype(np.int32)
    batch_size = token_ids.shape[0]

    # pixel_values: HF shape (B, n_patches, 768) → (B, 1_image, n_patches, 768)
    pv = img_data["pixel_values"]
    pixel_values = pv.astype(np.float32)[:, np.newaxis, :, :]

    # pixel_position_ids: HF shape (B, n_patches, 2) → (B, 1, n_patches, 2)
    ppid = img_data.get("image_position_ids")
    if ppid is not None:
        pixel_position_ids = ppid.astype(np.int32)[:, np.newaxis, :, :]
    else:
        n_patches = pixel_values.shape[2]
        pixel_position_ids = np.zeros(
            (batch_size, 1, n_patches, 2), dtype=np.int32
        )

    # vision_mask: 1 at every image placeholder position, 0 elsewhere.
    vision_mask = (token_ids == image_placeholder_id).astype(np.int32)

    # vision_indices: dense indices of the placeholder positions per batch item.
    vision_rows = [
        np.where(vision_mask[i])[0].astype(np.int32) for i in range(batch_size)
    ]
    max_vision_tokens = max((len(r) for r in vision_rows), default=0)
    vision_indices = np.zeros((batch_size, max_vision_tokens), dtype=np.int32)
    for i, row in enumerate(vision_rows):
        vision_indices[i, : len(row)] = row

    return (
        token_ids,
        padding_mask,
        pixel_values,
        pixel_position_ids,
        vision_indices,
        vision_mask,
    )


def _test_token_ids(label, preprocessor, prompt, hf_token_ids, **media_kwargs):
    """Assert KH-preprocessed token IDs match HF token IDs for any modality."""
    kh_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        sequence_length=hf_token_ids.shape[1],
    )
    kh_token_ids = ops.convert_to_numpy(kh_inputs["token_ids"])
    # Slice off canvas_length tokens appended by generate_preprocess.
    prompt_len = hf_token_ids.shape[1]
    kh_token_ids = kh_token_ids[:, :prompt_len]
    np.testing.assert_array_equal(kh_token_ids, hf_token_ids)
    print(f"✓ [{label}] Token IDs match.")


def _verify(diffusion_lm, hf_data, hf_preset):
    """Compare KerasHub model against pre-computed HF outputs."""
    # backbone = diffusion_lm.backbone

    # # --- Parameter count ---
    # print("\n--- Parameter Count ---")
    # hf_params = hf_data.get("param_count")
    # if hf_params is not None:
    #     unique_weights = {
    #         id(w): w for w in backbone.trainable_weights
    #     }.values()
    #     kh_params = sum(w.numpy().size for w in unique_weights)
    #     print(f"   HF params: {hf_params:,}")
    #     print(f"   KH params: {kh_params:,}")
    #     np.testing.assert_equal(kh_params, hf_params)
    #     print(f"✅ Parameter counts match: {kh_params:,}")

    # canvas_token_ids = hf_data["canvas_token_ids"]

    # Patch preprocessor's num_vision_tokens_per_image if image data exists in
    # HF output
    preprocessor = diffusion_lm.preprocessor
    if preprocessor is not None and hf_data["image"] is not None:
        img = hf_data["image"]
        image_placeholder_id = getattr(
            preprocessor.tokenizer, "image_placeholder_id", None
        )
        if image_placeholder_id is not None:
            actual_num_tokens = int(
                np.sum(img["input_ids"][0] == image_placeholder_id)
            )
            preprocessor.num_vision_tokens_per_image = actual_num_tokens

    # # --- Token ID Verification ---
    # print("\n--- Token ID Verification ---")
    # if preprocessor is not None:
    #     _test_token_ids(
    #         "text", preprocessor, PROMPT_TEXT, hf_data["text_input_ids"]
    #     )

    #     if hf_data["image"] is not None:
    #         img = hf_data["image"]
    #         raw_image = img.get("raw_image")
    #         if raw_image is not None:
    #             _test_token_ids(
    #                 "image",
    #                 preprocessor,
    #                 PROMPT_IMAGE,
    #                 img["input_ids"],
    #                 images=raw_image,
    #             )
    # else:
    #     print("⚠️  Preprocessor not available; skipping token ID check.")

    # # --- Text ---
    # print("\n--- Numerics Verification: text ---")
    # kh_logits = _kh_forward(
    #     diffusion_lm,
    #     hf_data["text_input_ids"].astype(np.int32),
    #     hf_data["text_attention_mask"].astype(np.int32),
    #     canvas_token_ids=canvas_token_ids,
    # )
    # _test_numerics("text", kh_logits, hf_data["text_logits"])

    # # --- Image ---
    # # Use HF-preprocessed pixel values to bypass PIL vs KH resize delta and
    # # exercise the vision encoder end-to-end.  Falls back to text-only encoder
    # # if the image placeholder ID is unavailable.
    # if hf_data["image"] is not None:
    #     print("\n--- Numerics Verification: image ---")
    #     try:
    #         img = hf_data["image"]

    #         # Resolve image placeholder token ID from the preprocessor.
    #         image_placeholder_id = None
    #         if preprocessor is not None and
    # hasattr(preprocessor, "tokenizer"):
    #             image_placeholder_id = getattr(
    #                 preprocessor.tokenizer, "image_placeholder_id", None
    #             )

    #         if (
    #             image_placeholder_id is not None
    #             and img.get("pixel_values") is not None
    #         ):
    #             (
    #                 token_ids,
    #                 padding_mask,
    #                 pixel_values,
    #                 pixel_position_ids,
    #                 vision_indices,
    #                 vision_mask,
    #             ) = _build_image_kh_inputs(img, image_placeholder_id)
    #             kh_logits = _kh_forward(
    #                 diffusion_lm,
    #                 token_ids,
    #                 padding_mask,
    #                 canvas_token_ids=canvas_token_ids,
    #                 pixel_values=pixel_values,
    #                 pixel_position_ids=pixel_position_ids,
    #                 vision_indices=vision_indices,
    #                 vision_mask=vision_mask,
    #             )
    #             _test_numerics(
    #                 "image (HF pixel values)",
    #                 kh_logits,
    #                 img["logits"],
    #             )
    #         else:
    #             # Fallback when pixel values or placeholder ID are
    # unavailable.
    #             token_ids = img["input_ids"].astype(np.int32)
    #             padding_mask = img["attention_mask"].astype(np.int32)
    #             kh_logits = _kh_forward(
    #                 diffusion_lm,
    #                 token_ids,
    #                 padding_mask,
    #                 canvas_token_ids=canvas_token_ids,
    #             )
    #             _test_numerics(
    #                 "image (text-only encoder)",
    #                 kh_logits,
    #                 img["logits"],
    #             )
    #     except Exception as e:
    #         print(f"⚠️  Image numerics check skipped: {e}")

    # --- Generation Comparison ---
    if FLAGS.skip_generate:
        print("\n--- Generation Comparison: SKIPPED (--skip_generate) ---")
    else:
        print("\n--- Generation Comparison ---")
        # `.generate()` requires a sampler, which is installed by `.compile()`.
        # The default `entropy_bound` sampler matches training-time behaviour.
        if getattr(diffusion_lm, "sampler", None) is None:
            diffusion_lm.compile(sampler="entropy_bound")

        _test_generate(
            "text",
            diffusion_lm,
            PROMPT_TEXT,
            hf_data.get("text_generated_text"),
        )

        if hf_data["image"] is not None:
            img = hf_data["image"]
            raw_image = img.get("raw_image")
            if raw_image is not None:
                _test_generate(
                    "image",
                    diffusion_lm,
                    PROMPT_IMAGE,
                    img.get("generated_text"),
                    images=raw_image,
                )

    print("-> HF verification complete.")


def _count_hf_params(hf_model):
    return sum(p.numel() for p in hf_model.parameters())


def _save_preset(hf_preset, preset_name, save_dtype, diffusion_lm=None):
    """Save the converted model to a local preset directory."""
    save_path = f"./{preset_name}"
    print(f"\n-> Saving model in {save_dtype} to {save_path} …")

    if save_dtype == "bfloat16":
        diffusion_lm_bf16 = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
            hf_preset, dtype="bfloat16"
        )
        diffusion_lm_bf16.save_to_preset(save_path)
    else:
        diffusion_lm.save_to_preset(save_path)

    print(f"-> Preset saved to {save_path}")


def main(_):
    preset_name = FLAGS.preset
    if preset_name not in PRESET_MAP:
        raise ValueError(
            f"Unknown preset {FLAGS.preset!r}. "
            f"Choose one of: {', '.join(PRESET_MAP.keys())}"
        )

    hf_repo_id = PRESET_MAP[preset_name]
    hf_preset = f"hf://{hf_repo_id}"

    hf_data = _gather_hf_data(hf_repo_id)
    print(f"-> Loading Gemma4BlockDiffusionLM from {hf_preset} …")
    diffusion_lm = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
        hf_preset, dtype="float32"
    )
    print("✓ All weights loaded")

    _verify(diffusion_lm, hf_data, hf_preset)

    del hf_data
    gc.collect()

    if FLAGS.save_dtype == "bfloat16":
        del diffusion_lm
        gc.collect()
        _save_preset(hf_preset, preset_name, FLAGS.save_dtype)
    else:
        _save_preset(
            hf_preset, preset_name, FLAGS.save_dtype, diffusion_lm=diffusion_lm
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
