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

try:
    import librosa
except ImportError:
    raise ImportError(
        "Gemma3n checkpoint conversion audio tests require librosa. "
        "Please install it via `pip install librosa`."
    )

PRESET_MAP = {
    "gemma3n_e2b": "google/gemma-3n-E2B",
    "gemma3n_e2b_it": "google/gemma-3n-E2B-it",
    "gemma3n_e4b": "google/gemma-3n-E4B",
    "gemma3n_e4b_it": "google/gemma-3n-E4B-it",
}
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)
flags.mark_flag_as_required("preset")


_TEST_WAV_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../keras_hub/src/tests/test_data/audio_transcription_tests/"
        "female_short_voice_clip_17sec.wav",
    )
)


def _load_test_audio_waveform(hf_processor):
    if not os.path.exists(_TEST_WAV_PATH):
        raise FileNotFoundError(
            f"Expected test WAV file not found at: {_TEST_WAV_PATH}"
        )
    sampling_rate = hf_processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(_TEST_WAV_PATH, sr=sampling_rate, mono=True)
    return audio.astype(np.float32)


def validate_output(keras_model, keras_preprocessor, hf_model, hf_processor):
    keras_params = keras_model.count_params()
    # count tied embeddings only once for the comparison
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
    text = "What is Keras?"
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

    hf_seq_len = text_inputs["input_ids"].shape[1]
    preprocessed_text_inputs = keras_preprocessor.generate_preprocess(
        [text], sequence_length=hf_seq_len
    )

    print("  -> Running Keras model forward pass...")
    keras_output = keras_model(preprocessed_text_inputs)
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

    mm_prompt_kh = (
        "A cat sat on a mat"
        f"{keras_preprocessor.start_of_image_token}"
        "<end_of_turn>\n"
        f"{keras_preprocessor.start_of_audio_token}"
    )
    mm_backbone_inputs = keras_preprocessor.generate_preprocess(
        {
            "prompts": [mm_prompt_kh],
            "images": [np.asarray(image)],
            "audios": [audio_data],
        },
        sequence_length=int(hf_mm_inputs["input_ids"].shape[1]),
    )
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


def validate_generate(keras_causal_lm, hf_model, hf_processor):
    prompt_kh = (
        "<start_of_turn>user\n"
        "Transcribe this audio: <start_of_audio>"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    prompt_hf = (
        f"<start_of_turn>user\n"
        f"Transcribe this audio: {hf_processor.audio_token}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    audio_data = _load_test_audio_waveform(hf_processor)

    keras_causal_lm.compile(sampler="greedy")
    keras_text = keras_causal_lm.generate(
        {"prompts": prompt_kh, "audios": audio_data},
        max_length=256,
        strip_prompt=True,
    )

    hf_inputs = hf_processor(
        text=prompt_hf,
        audio=[audio_data],
        return_tensors="pt",
        padding="longest",
    )

    hf_outputs = hf_model.generate(
        **hf_inputs,
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
    )
    hf_prompt_len = hf_inputs["input_ids"].shape[1]
    hf_new_tokens = hf_outputs[:, hf_prompt_len:]
    hf_text = hf_processor.batch_decode(
        hf_new_tokens, skip_special_tokens=True
    )[0]

    print("🔶 Keras generate output:", keras_text)
    print("🔶 HuggingFace generate output:", hf_text)
    print(
        "🔶 Generation match:",
        "✅ Yes" if keras_text == hf_text else "⚠️ No",
    )


def _load_hf_model_and_processor(hf_model_name, torch_dtype):
    """Load (or download) the HF model and processor."""
    print(f"  -> Downloading Hugging Face model: {hf_model_name}")
    hf_model = Gemma3nForConditionalGeneration.from_pretrained(
        hf_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    hf_processor = Gemma3nProcessor.from_pretrained(hf_model_name)
    hf_model.eval()
    return hf_model, hf_processor


def main(_):
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    save_path = preset
    save_dtype = FLAGS.save_dtype

    print("=" * 60)
    print("  FLOAT32 VALIDATION")
    print("=" * 60)
    hf_model, hf_processor = _load_hf_model_and_processor(
        hf_model_name,
        torch.float32,
    )
    print("-> Loading Keras model (float32) from HuggingFace preset.")
    keras_preprocessor = (
        keras_hub.models.Gemma3nCausalLMPreprocessor.from_preset(
            f"hf://{hf_model_name}"
        )
    )
    keras_model = keras_hub.models.Gemma3nBackbone.from_preset(
        f"hf://{hf_model_name}", dtype="float32"
    )
    print("\n-> Validating output consistency (float32).")
    validate_output(keras_model, keras_preprocessor, hf_model, hf_processor)

    keras_causal_lm = keras_hub.models.Gemma3nCausalLM(
        backbone=keras_model,
        preprocessor=keras_preprocessor,
    )
    if hf_model_name.endswith("-it"):
        # Audio input only works well with instruction-tuned models.
        print("\n-> Validating generation consistency (float32).")
        validate_generate(keras_causal_lm, hf_model, hf_processor)

    print("\n" + "=" * 60)
    print("  SAVING PRESET")
    print("=" * 60)
    if save_dtype == "float32":
        # Already validated in float32 — save directly.
        print(f"-> Saving Keras preset ({save_dtype}) to ./{save_path}")
        keras_causal_lm.save_to_preset(f"./{save_path}")
    else:
        # Free memory before reloading in the target save dtype.
        del keras_causal_lm, keras_model, hf_model
        gc.collect()

        print(f"-> Loading Keras model ({save_dtype}) from HuggingFace preset.")
        keras_model_save = keras_hub.models.Gemma3nBackbone.from_preset(
            f"hf://{hf_model_name}", dtype=save_dtype
        )
        keras_causal_lm_save = keras_hub.models.Gemma3nCausalLM(
            backbone=keras_model_save,
            preprocessor=keras_preprocessor,
        )
        print(f"💾 Saving Keras preset ({save_dtype}) to ./{save_path}")
        keras_causal_lm_save.save_to_preset(f"./{save_path}")

    print("🎉 Conversion complete.")


if __name__ == "__main__":
    app.run(main)
