import gc
import os

import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import Gemma3nForConditionalGeneration
from transformers import Gemma3nProcessor

import keras_hub

PRESET_MAP = {
    "gemma3n_e2b": "google/gemma-3n-E2B",
}
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "cache_dir", "./hf_cache", "Directory to cache Hugging Face downloads."
)
flags.mark_flag_as_required("preset")


def validate_output(keras_model, hf_model, hf_processor):
    keras_params = keras_model.count_params()
    hf_params = sum(
        p.numel() for n, p in hf_model.named_parameters() if "lm_head" not in n
    )
    print("🔶 Parameter count comparison:")
    print(f"   -> KerasHub backbone:    {keras_params:,}")
    print(f"   -> Huggingface backbone: {hf_params:,}")
    if keras_params != hf_params:
        print("⚠️ Parameter counts do not match!")
    else:
        print("✅ Parameter counts match.")

    print("🔶 Validating model outputs (text-only)...")
    text = (
        " This gap is likely due to error accumulation across 30 complex "
        "layers (AltUp, Laurel, per-layer inputs) between JAX and "
        "PyTorch backends."
    )
    text_inputs = hf_processor(
        text=text,
        return_tensors="pt",
        padding=False,
    )
    print(f"  -> Input tokens: {text_inputs['input_ids'][0].tolist()}")
    print("  -> Running HF model forward pass...")
    hf_hidden = hf_model.model(**text_inputs).last_hidden_state
    hf_output = hf_model.lm_head(hf_hidden)
    final_logit_soft_cap = getattr(
        hf_model.config.get_text_config(),
        "final_logit_softcapping",
        None,
    )
    if final_logit_soft_cap is not None:
        hf_output = hf_output / final_logit_soft_cap
        hf_output = torch.tanh(hf_output)
        hf_output = hf_output * final_logit_soft_cap
    hf_output = hf_output.detach().cpu().float().numpy()
    print(f"  -> HF output shape: {hf_output.shape}")
    img_size = hf_processor.image_processor.size
    dummy_image = np.zeros(
        (1, 1, img_size["height"], img_size["width"], 3),
        dtype=np.float32,
    )
    dummy_audio = np.zeros(
        (1, 1, 1, 128),
        dtype=np.float32,
    )
    dummy_audio_mask = np.zeros(
        (1, 1, 1),
        dtype=bool,
    )
    backbone_keras_inputs = {
        "token_ids": text_inputs["input_ids"].numpy(),
        "padding_mask": text_inputs["attention_mask"].numpy().astype(bool),
        "images": dummy_image,
        "input_features": dummy_audio,
        "input_features_mask": dummy_audio_mask,
    }
    print("  -> Running Keras model forward pass...")
    keras_output = keras_model(backbone_keras_inputs)
    keras_output = keras_model.language_model.token_embedding(
        keras_output, reverse=True
    )
    print(f"  -> Keras output shape: {keras_output.shape}")
    keras_output = np.array(keras_output)
    abs_diff = np.abs(keras_output - hf_output)
    mean_diff = float(np.mean(abs_diff))
    print(f"🔶 Mean absolute difference: {mean_diff:.8f}")
    hf_pred_tokens = np.argmax(hf_output[0], axis=-1)
    keras_pred_tokens = np.argmax(keras_output[0], axis=-1)
    print(f"  -> HF predicted tokens:    {hf_pred_tokens.tolist()}")
    print(f"  -> Keras predicted tokens: {keras_pred_tokens.tolist()}")
    tokens_match = np.array_equal(hf_pred_tokens, keras_pred_tokens)
    print(
        f"  -> Predicted tokens match: {'✅ Yes' if tokens_match else '⚠️ No'}"
    )
    print(f"  -> HF logits[0,:5]:    {hf_output[0, 0, :5]}")
    print(f"  -> Keras logits[0,:5]: {keras_output[0, 0, :5]}")

    try:
        np.testing.assert_allclose(
            keras_output,
            hf_output,
            atol=1e-4,
            rtol=1e-4,
        )
        print("✅ Text-only logits within 1e-4 tolerance.")
    except AssertionError as err:
        print(err.args[0])

    print(f"\n{'=' * 60}")
    print("  MULTIMODAL VALIDATION (text + image + audio)")
    print(f"{'=' * 60}")
    image_size = hf_processor.image_processor.size
    image = Image.new(
        "RGB",
        (image_size["width"], image_size["height"]),
    )
    sampling_rate = hf_processor.feature_extractor.sampling_rate
    audio_data = np.zeros(int(sampling_rate * 2.0))
    mm_text = (
        f"A cat sat on a mat"
        f"{hf_processor.image_token}"
        f"<end_of_turn>\n"
        f"{hf_processor.audio_token}"
    )
    hf_mm_inputs = hf_processor(
        text=mm_text,
        images=image,
        audio=[audio_data],
        return_tensors="pt",
        padding="longest",
    )
    print(f"  -> Num tokens: {hf_mm_inputs['input_ids'].shape[1]}")
    print("  -> Running HF multimodal forward pass...")
    with torch.no_grad():
        hf_mm_hidden = hf_model.model(**hf_mm_inputs).last_hidden_state
        hf_mm_output = hf_model.lm_head(hf_mm_hidden)
        if final_logit_soft_cap is not None:
            hf_mm_output = hf_mm_output / final_logit_soft_cap
            hf_mm_output = torch.tanh(hf_mm_output)
            hf_mm_output = hf_mm_output * final_logit_soft_cap
    hf_mm_output = hf_mm_output.detach().cpu().float().numpy()
    print(f"  -> HF output shape: {hf_mm_output.shape}")
    mm_keras_inputs = {k: v.numpy() for k, v in hf_mm_inputs.items()}
    mm_backbone_inputs = {}
    mm_backbone_inputs["token_ids"] = mm_keras_inputs.pop("input_ids")
    mm_backbone_inputs["padding_mask"] = mm_keras_inputs.pop(
        "attention_mask"
    ).astype(bool)
    pixel_values = mm_keras_inputs.pop("pixel_values")
    pixel_values_t = np.transpose(pixel_values, (0, 2, 3, 1))
    if pixel_values_t.ndim == 4:
        pixel_values_t = np.expand_dims(pixel_values_t, axis=1)
    mm_backbone_inputs["images"] = pixel_values_t
    input_features = mm_keras_inputs.pop("input_features")
    input_features_mask = mm_keras_inputs.pop("input_features_mask")
    input_features_mask = ~input_features_mask
    if input_features.ndim == 3:
        input_features = np.expand_dims(input_features, axis=1)
    if input_features_mask.ndim == 2:
        input_features_mask = np.expand_dims(input_features_mask, axis=1)
    mm_backbone_inputs["input_features"] = input_features
    mm_backbone_inputs["input_features_mask"] = input_features_mask
    print("  -> Running Keras multimodal forward pass...")
    keras_mm_output = keras_model(mm_backbone_inputs)
    keras_mm_output = keras_model.language_model.token_embedding(
        keras_mm_output, reverse=True
    )
    keras_mm_output = np.array(keras_mm_output)
    mm_abs_diff = np.abs(keras_mm_output - hf_mm_output)
    mm_mean = float(np.mean(mm_abs_diff))
    print(f"🔶 Mean absolute difference: {mm_mean:.8f}")
    hf_mm_pred = np.argmax(hf_mm_output[0], axis=-1)
    keras_mm_pred = np.argmax(keras_mm_output[0], axis=-1)
    mm_tokens_match = np.array_equal(hf_mm_pred, keras_mm_pred)
    match_count = np.sum(hf_mm_pred == keras_mm_pred)
    total_count = len(hf_mm_pred)
    print(
        f"  -> Predicted tokens match: "
        f"{match_count}/{total_count} "
        f"({'✅' if mm_tokens_match else '⚠️'})"
    )
    print(f"  -> HF logits[0,:5]:    {hf_mm_output[0, 0, :5]}")
    print(f"  -> Keras logits[0,:5]: {keras_mm_output[0, 0, :5]}")
    input_ids = mm_backbone_inputs["token_ids"][0]
    vision_offset = keras_model.embed_vision.vocab_offset
    audio_offset = keras_model.embed_audio.vocab_offset
    text_pos = np.where(input_ids < vision_offset)[0]
    vision_pos = np.where(
        (input_ids >= vision_offset) & (input_ids < audio_offset)
    )[0]
    audio_pos = np.where(input_ids >= audio_offset)[0]
    for label, positions in [
        ("Text", text_pos),
        ("Vision", vision_pos),
        ("Audio", audio_pos),
    ]:
        if len(positions) > 0:
            mod_diff = mm_abs_diff[0, positions, :]
            print(
                f"  -> {label} tokens "
                f"({len(positions)} pos): "
                f"mean={np.mean(mod_diff):.8f}, "
                f"max={np.max(mod_diff):.8f}"
            )
    try:
        np.testing.assert_allclose(
            keras_mm_output,
            hf_mm_output,
            atol=1e-4,
            rtol=1e-4,
        )
        print("✅ Multimodal logits within 1e-4 tolerance.")
    except AssertionError as err:
        print(err.args[0])


def _load_hf_model_and_processor(
    preset,
    hf_model_name,
    cache_dir,
    torch_dtype,
):
    """Load (or download) the HF model and processor."""
    model_cache_path = os.path.join(cache_dir, f"{preset}_model")
    processor_cache_path = os.path.join(cache_dir, f"{preset}_processor")
    hf_model = None
    hf_processor = None
    if os.path.exists(model_cache_path) and os.path.exists(
        processor_cache_path
    ):
        print(
            "  -> Loading cached Hugging Face model and processor"
            f" from {cache_dir}"
        )
        try:
            hf_model = Gemma3nForConditionalGeneration.from_pretrained(
                model_cache_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            hf_processor = Gemma3nProcessor.from_pretrained(
                processor_cache_path
            )
        except Exception as e:
            print(f"⚠️ Failed to load from cache: {e}. Downloading again...")
            hf_model = None
            hf_processor = None
    if hf_model is None or hf_processor is None:
        print(f"  -> Downloading Hugging Face model: {hf_model_name}")
        hf_model = Gemma3nForConditionalGeneration.from_pretrained(
            hf_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        hf_processor = Gemma3nProcessor.from_pretrained(hf_model_name)
        print(f"💾 Saving model and processor to cache: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        hf_model.save_pretrained(model_cache_path)
        hf_processor.save_pretrained(processor_cache_path)
    hf_model.eval()
    return hf_model, hf_processor


def main(_):
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    cache_dir = FLAGS.cache_dir
    save_path = preset

    print("=" * 60)
    print("  FLOAT32 VALIDATION")
    print("=" * 60)
    hf_model_f32, hf_processor = _load_hf_model_and_processor(
        preset,
        hf_model_name,
        cache_dir,
        torch.float32,
    )
    print("-> Loading Keras model (float32) from HuggingFace preset.")
    keras_model_f32 = keras_hub.models.Gemma3nBackbone.from_preset(
        f"hf://{hf_model_name}", dtype="float32"
    )
    print("\n-> Validating output consistency (float32).")
    validate_output(keras_model_f32, hf_model_f32, hf_processor)
    del keras_model_f32, hf_model_f32
    gc.collect()

    print("\n" + "=" * 60)
    print("  SAVING BFLOAT16 PRESET")
    print("=" * 60)
    print("-> Loading Keras model (bfloat16) from HuggingFace preset.")
    keras_model_bf16 = keras_hub.models.Gemma3nBackbone.from_preset(
        f"hf://{hf_model_name}", dtype="bfloat16"
    )
    print(f"💾 Saving Keras preset to ./{save_path}")
    keras_model_bf16.save_to_preset(f"./{save_path}")
    print("🎉 Conversion complete.")
    del keras_model_bf16
    gc.collect()


if __name__ == "__main__":
    app.run(main)
