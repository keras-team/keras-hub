"""Convert DiffusionGemma HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_diffusion_gemma_checkpoints.py \
        --preset diffusion_gemma_26b_a4b_it \
        --save_dtype bfloat16
"""

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
from transformers import AutoProcessor
from transformers import DiffusionGemmaForBlockDiffusion
from transformers import DynamicCache

import keras_hub

GENERATION_SEED = 123
CANVAS_SEED = 1234
MAX_GENERATE_TOKENS = 32
torch.manual_seed(GENERATION_SEED)

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


def _count_hf_params(hf_model):
    return sum(parameter.numel() for parameter in hf_model.parameters())


def _precompute_hf_outputs(hf_repo_id):
    """Load HF model, run all HF-side computations, return a data dict."""
    hf_model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        hf_repo_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(hf_repo_id)

    is_multimodal = hf_model.config.vision_config is not None

    param_count = _count_hf_params(hf_model)

    canvas_length = hf_model.config.canvas_length
    vocab_size = len(processor.tokenizer)

    # The text check uses a seeded random canvas. Diverse token embeddings keep
    # float32 rounding away from an MoE routing boundary at layer 29.
    # The image check uses an all-zeros canvas.
    rng = np.random.default_rng(CANVAS_SEED)
    text_canvas_np = rng.integers(
        0, vocab_size, size=(1, canvas_length), dtype=np.int32
    )
    text_canvas = torch.tensor(text_canvas_np, dtype=torch.long)
    image_canvas_np = np.zeros((1, canvas_length), dtype=np.int32)
    image_canvas = torch.tensor(image_canvas_np, dtype=torch.long)

    text_fwd = _hf_forward(
        hf_model, processor, PROMPT_TEXT, decoder_input_ids=text_canvas
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
                decoder_input_ids=image_canvas,
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
            print(f"! Image HF forward skipped: {e}")

    del hf_model, processor
    gc.collect()

    return {
        "param_count": param_count,
        "text_canvas_token_ids": text_canvas_np,
        "image_canvas_token_ids": image_canvas_np,
        "text_logits": text_fwd["logits"],
        "text_input_ids": text_fwd["input_ids"],
        "text_attention_mask": text_fwd["attention_mask"],
        "text_generated_text": text_fwd["generated_text"],
        "image": image_data,
    }


def _load_test_image():
    print(f"   Downloading verification image from {IMAGE_URL} ...", flush=True)
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def _hf_forward(
    hf_model,
    processor,
    prompt,
    raw_image=None,
    decoder_input_ids=None,
):
    """Run one HF forward pass and return decoder logits and generated text.

    Always pass a fixed ``decoder_input_ids`` canvas. Without it, the model
    samples a random canvas and the check is not repeatable.
    """
    proc_kwargs = {"text": prompt, "return_tensors": "pt"}
    if raw_image is not None:
        proc_kwargs["images"] = raw_image
    hf_inputs = processor(**proc_kwargs)
    hf_inputs = {k: v.cpu() for k, v in hf_inputs.items()}

    # The KerasHub preprocessor adds a BOS token. The prompts have no BOS
    # marker. Add BOS to the HF inputs so both sides see the same sequence.
    bos_id = processor.tokenizer.bos_token_id
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
        # `mm_token_type_ids` has one entry per token. Shift it by the BOS
        # token. Otherwise the vision-bidirectional mask is off by one.
        if "mm_token_type_ids" in hf_inputs:
            text_type = torch.zeros(
                (hf_inputs["mm_token_type_ids"].shape[0], 1),
                dtype=hf_inputs["mm_token_type_ids"].dtype,
                device=hf_inputs["mm_token_type_ids"].device,
            )
            hf_inputs["mm_token_type_ids"] = torch.cat(
                [text_type, hf_inputs["mm_token_type_ids"]], dim=1
            )

    forward_inputs = dict(hf_inputs)
    if decoder_input_ids is not None:
        forward_inputs["decoder_input_ids"] = decoder_input_ids.cpu()

    # `DiffusionGemmaEncoderModel.forward()` ignores its own mask when
    # `attention_mask` is not a dict. The caller must build the real mask,
    # including the vision-bidirectional mask. `generate()` does this.
    # Build the mask here so the reference matches `generate()`.
    input_ids = forward_inputs["input_ids"]
    attention_mask_tensor = forward_inputs.get(
        "attention_mask", torch.ones_like(input_ids)
    )
    batch_size, seq_len = input_ids.shape
    dummy_input_embeds = torch.empty(
        (batch_size, seq_len, 0), dtype=hf_model.dtype, device=input_ids.device
    )
    encoder_position_ids = (
        torch.arange(seq_len, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    # `generate()` passes an empty `DynamicCache`, not `None`. Pass the same
    # empty cache so the mask code takes the same branch.
    encoder_cache = DynamicCache(
        config=hf_model.config.get_text_config(decoder=True)
    )
    forward_inputs["attention_mask"] = (
        hf_model.model.encoder.create_masks_for_generate(
            config=hf_model.config,
            inputs_embeds=dummy_input_embeds,
            attention_mask=attention_mask_tensor,
            past_key_values=encoder_cache,
            position_ids=encoder_position_ids,
            mm_token_type_ids=forward_inputs.get("mm_token_type_ids"),
        )
    )

    with torch.no_grad():
        hf_out = hf_model(**forward_inputs, output_hidden_states=False)

    logits = hf_out.logits.detach().cpu().float().numpy()
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
        # One short canvas makes generation faster. The numerics checks keep
        # the full canvas.
        canvas_length = hf_model.config.canvas_length
        hf_model.config.canvas_length = MAX_GENERATE_TOKENS
        try:
            with torch.no_grad():
                output = hf_model.generate(
                    **hf_inputs, max_new_tokens=MAX_GENERATE_TOKENS
                )
            sequence = output[0]
            if sequence.ndim > 1:
                sequence = sequence[0]
            prompt_length = hf_inputs["input_ids"].shape[-1]
            # HF returns the whole last canvas. Slice it to the length of the
            # KerasHub output.
            generated_text = processor.decode(
                sequence[prompt_length : prompt_length + MAX_GENERATE_TOKENS],
                skip_special_tokens=True,
            )
        except Exception as e:
            print(f"✗ HF .generate() failed ({e}).")
        finally:
            hf_model.config.canvas_length = canvas_length

    return {
        "logits": logits,
        "input_ids": hf_inputs["input_ids"].numpy(),
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

    ``canvas_token_ids`` must match the ``decoder_input_ids`` passed to the
    HF model.
    """
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    if pixel_values is not None:
        inputs["pixel_values"] = pixel_values
        inputs["pixel_position_ids"] = pixel_position_ids
        inputs["vision_indices"] = vision_indices
        inputs["vision_mask"] = vision_mask

    with torch.no_grad():
        # Step 1: encoder — builds KV cache over the prompt.
        encoder_kv_cache, prompt_length = diffusion_lm._encode_prompt(inputs)

        # Step 2: canvas embeddings (first step → self-conditioning is no-op).
        canvas_embeds = diffusion_lm._prepare_canvas_embeds(
            canvas_token_ids, prev_logits=None
        )

        # Step 3: decoder — one bidirectional denoising step.
        hidden = diffusion_lm._decode_canvas_step(
            canvas_embeds,
            encoder_kv_cache,
            prompt_length,
            prompt_padding_mask=padding_mask,
        )

        # Step 4: project to vocabulary logits with soft-cap.
        logits = diffusion_lm._canvas_logits(hidden)

    return ops.convert_to_numpy(logits).astype(np.float32)


def _report_numerics(label, kh_logits, hf_logits):
    """Log max/mean absolute logit difference and matching %; warn if > 1e-3."""
    # Trim to common sequence length.
    min_len = min(kh_logits.shape[1], hf_logits.shape[1])
    kh = kh_logits[:, :min_len, :]
    hf = hf_logits[:, :min_len, :]

    abs_diff = np.abs(kh - hf)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    # Fraction of logits inside the same band `assert_allclose` checks below.
    tol = 1e-3 + 1e-3 * np.abs(hf)
    total = hf.size
    matched = total - int(np.sum(abs_diff > tol))
    pct = 100.0 * matched / total

    try:
        np.testing.assert_allclose(kh, hf, atol=1e-3, rtol=1e-3)
        print(
            f"✓ [{label}] Logits within 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={pct:.2f}% ({matched}/{total})."
        )
    except AssertionError:
        print(
            f"✗ [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={pct:.2f}% ({matched}/{total})."
        )


def _test_generate(label, diffusion_lm, prompt, hf_generated_text, images=None):
    """Run KH ``.generate()`` and print the output alongside HF's output."""
    if hf_generated_text is None:
        print(
            f"[{label}] Generate comparison skipped: HF generation unavailable."
        )
        return

    diffusion_lm.compile(
        sampler=keras_hub.samplers.EntropyBoundSampler(seed=GENERATION_SEED)
    )

    kh_inputs = prompt
    if images is not None:
        kh_inputs = {"prompts": prompt, "images": [images]}
    # Use the same short canvas as the HF side. Restore the value afterwards,
    # because `save_to_preset` serializes `canvas_length`.
    canvas_length = diffusion_lm.canvas_length
    diffusion_lm.canvas_length = MAX_GENERATE_TOKENS
    try:
        kh_output = diffusion_lm.generate(
            kh_inputs, max_length=MAX_GENERATE_TOKENS
        )
    except Exception as e:
        print(f"✗ [{label}] KH .generate() failed: {e}")
        return
    finally:
        diffusion_lm.canvas_length = canvas_length
    kh_text = kh_output[0] if isinstance(kh_output, list) else kh_output

    if images is not None:
        print(
            f"\n[{label}] Input image: url={IMAGE_URL}, "
            f"size={images.size}, mode={images.mode}"
        )
    print(f"\n[{label}] HF generate output:\n  {hf_generated_text}")
    print(f"[{label}] KH generate output:\n  {kh_text}")


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
        dict of numpy arrays. The keys match the arguments of ``_kh_forward``.
    """
    token_ids = img_data["input_ids"].astype(np.int32)
    batch_size = token_ids.shape[0]

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

    return {
        "token_ids": token_ids,
        "padding_mask": img_data["attention_mask"].astype(np.int32),
        # HF shape (B, n_patches, 768) -> (B, 1_image, n_patches, 768)
        "pixel_values": img_data["pixel_values"].astype(np.float32)[
            :, np.newaxis, :, :
        ],
        # HF shape (B, n_patches, 2) -> (B, 1, n_patches, 2)
        "pixel_position_ids": img_data["image_position_ids"].astype(np.int32)[
            :, np.newaxis, :, :
        ],
        "vision_indices": vision_indices,
        "vision_mask": vision_mask,
    }


def _test_numerics(diffusion_lm, hf_data, modality):
    """Run the KH forward pass for one modality and report it against HF.

    Args:
        diffusion_lm: The KerasHub model.
        hf_data: dict. The pre-computed HF outputs.
        modality: str. Either ``"text"`` or ``"image"``. The image check uses
            the HF pixel values. This removes the PIL-vs-KerasHub resize
            difference.
    """
    if modality == "image":
        tokenizer = diffusion_lm.preprocessor.tokenizer
        inputs = _build_image_kh_inputs(
            hf_data["image"], tokenizer.image_placeholder_id
        )
        label = "image"
        hf_logits = hf_data["image"]["logits"]
    else:
        inputs = {
            "token_ids": hf_data["text_input_ids"].astype(np.int32),
            "padding_mask": hf_data["text_attention_mask"].astype(np.int32),
        }
        label = "text"
        hf_logits = hf_data["text_logits"]

    kh_logits = _kh_forward(
        diffusion_lm,
        canvas_token_ids=hf_data[f"{modality}_canvas_token_ids"],
        **inputs,
    )
    _report_numerics(label, kh_logits, hf_logits)


def _test_parameter_count(backbone, hf_params):
    """Assert the KerasHub backbone has the same parameter count as HF."""
    unique_weights = {id(w): w for w in backbone.trainable_weights}.values()
    kh_params = sum(int(np.prod(weight.shape)) for weight in unique_weights)
    print(f"   HF params: {hf_params:,}")
    print(f"   KH params: {kh_params:,}")
    np.testing.assert_equal(kh_params, hf_params)
    print(f"✓ Parameter counts match: {kh_params:,}")


def _test_token_ids(label, preprocessor, prompt, hf_token_ids, **media_kwargs):
    """Assert KH-preprocessed token IDs match HF token IDs for any modality."""
    kh_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        sequence_length=hf_token_ids.shape[1],
    )
    kh_token_ids = ops.convert_to_numpy(kh_inputs["token_ids"])
    np.testing.assert_array_equal(kh_token_ids, hf_token_ids)
    print(f"✓ [{label}] Token IDs match.")


def _verify(diffusion_lm, hf_data):
    """Compare KerasHub model against pre-computed HF outputs."""
    image = hf_data["image"]

    print("\n--- Parameter Count ---")
    _test_parameter_count(diffusion_lm.backbone, hf_data["param_count"])

    preprocessor = diffusion_lm.preprocessor

    # --- Token ID Verification ---
    print("\n--- Token ID Verification ---")
    _test_token_ids(
        "text", preprocessor, PROMPT_TEXT, hf_data["text_input_ids"]
    )
    if image is not None:
        _test_token_ids(
            "image",
            preprocessor,
            PROMPT_IMAGE,
            image["input_ids"],
            images=image["raw_image"],
        )

    # --- Numerics Verification ---
    print("\n--- Numerics Verification ---")
    _test_numerics(diffusion_lm, hf_data, "text")
    if image is not None:
        _test_numerics(diffusion_lm, hf_data, "image")

    # --- Generation Comparison ---
    if FLAGS.skip_generate:
        print("\n--- Generation Comparison: SKIPPED (--skip_generate) ---")
        return

    print("\n--- Generation Comparison ---")
    _test_generate(
        "text",
        diffusion_lm,
        PROMPT_TEXT,
        hf_data["text_generated_text"],
    )
    if image is not None:
        _test_generate(
            "image",
            diffusion_lm,
            PROMPT_IMAGE,
            image["generated_text"],
            images=image["raw_image"],
        )


def _save_preset(hf_preset, preset_name, save_dtype, diffusion_lm=None):
    """Save the converted model to a local preset directory."""
    save_path = f"./{preset_name}"
    print(f"\n-> Saving model in {save_dtype} to {save_path} …")

    if save_dtype == "bfloat16":
        diffusion_lm_bf16 = (
            keras_hub.models.DiffusionGemmaBlockDiffusionLM.from_preset(
                hf_preset, dtype="bfloat16"
            )
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

    hf_data = _precompute_hf_outputs(hf_repo_id)
    diffusion_lm = keras_hub.models.DiffusionGemmaBlockDiffusionLM.from_preset(
        hf_preset, dtype="float32"
    )
    print("✓ All weights loaded")
    _verify(diffusion_lm, hf_data)

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
