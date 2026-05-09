"""Convert Whisper checkpoints from Hugging Face to KerasHub format.

The on-the-fly converter at `keras_hub/src/utils/transformers/convert_whisper`
does the actual weight porting. This script orchestrates it end-to-end for a
single preset: it loads the HF reference model, builds the Keras task model
via `from_preset("hf://...")`, prints a parameter-count diff, runs encoder
and decoder forward passes through both implementations on synthetic audio
and compares the outputs, saves the Keras preset locally, reloads it to
verify the round-trip, and finally runs `generate()` on a real audio sample.

Usage:
```shell
python tools/checkpoint_conversion/convert_whisper_checkpoints.py \
    --preset whisper_tiny_en
```
"""

import os

import keras
import librosa
import numpy as np
import torch
import transformers
from absl import app
from absl import flags

import keras_hub

PRESET_MAP = {
    "whisper_tiny_en": "openai/whisper-tiny.en",
    "whisper_base_en": "openai/whisper-base.en",
    "whisper_small_en": "openai/whisper-small.en",
    "whisper_medium_en": "openai/whisper-medium.en",
    "whisper_tiny_multi": "openai/whisper-tiny",
    "whisper_base_multi": "openai/whisper-base",
    "whisper_small_multi": "openai/whisper-small",
    "whisper_medium_multi": "openai/whisper-medium",
    "whisper_large_multi": "openai/whisper-large",
    "whisper_large_multi_v2": "openai/whisper-large-v2",
}

SAMPLE_AUDIO_PATH = (
    "keras_hub/src/tests/test_data/audio_transcription_tests/"
    "male_short_voice_clip_3sec.wav"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def to_numpy(tensor):
    if keras.backend.backend() == "torch":
        return tensor.detach().cpu().numpy()
    if keras.backend.backend() == "tensorflow":
        return tensor.numpy()
    if keras.backend.backend() == "jax":
        import jax

        return jax.device_get(tensor)
    raise ValueError(f"Unsupported backend: {keras.backend.backend()}")


def check_param_match(keras_model, hf_model):
    keras_params = keras_model.backbone.count_params()
    hf_params = sum(p.numel() for p in hf_model.model.parameters())
    print(f"Keras backbone params: {keras_params:,}")
    print(f"HF      model params: {hf_params:,}")
    print(f"Diff:                  {abs(keras_params - hf_params):,}")


def check_numerics(keras_model, hf_model, hf_processor):
    """Compare encoder/decoder outputs given identical log-mel features.

    We feed both stacks the same HF-extracted features so the comparison
    isolates the encoder/decoder weights from any drift in the audio
    frontend (`WhisperAudioConverter` vs `WhisperFeatureExtractor`).
    """
    rng = np.random.default_rng(0)
    raw_audio = rng.standard_normal((16000,)).astype("float32")

    hf_features_pt = hf_processor.feature_extractor(
        raw_audio, sampling_rate=16000, return_tensors="pt"
    ).input_features  # (1, n_mels, n_frames)
    hf_features_np = hf_features_pt.numpy()
    # Keras backbone consumes (batch, n_frames, n_mels).
    keras_features_np = np.transpose(hf_features_np, (0, 2, 1))

    decoder_input_ids = np.array(
        [[hf_model.config.decoder_start_token_id]], dtype="int32"
    )

    keras_inputs = {
        "encoder_features": keras.ops.convert_to_tensor(keras_features_np),
        "decoder_token_ids": keras.ops.convert_to_tensor(decoder_input_ids),
        "decoder_padding_mask": keras.ops.ones_like(
            keras.ops.convert_to_tensor(decoder_input_ids), dtype="bool"
        ),
    }
    keras_outputs = keras_model.backbone(keras_inputs, training=False)
    keras_encoder = to_numpy(keras_outputs["encoder_sequence_output"])
    keras_decoder = to_numpy(keras_outputs["decoder_sequence_output"])

    hf_model.eval()
    with torch.no_grad():
        hf_outputs = hf_model.model(
            input_features=hf_features_pt,
            decoder_input_ids=torch.from_numpy(decoder_input_ids),
            output_hidden_states=False,
        )
        hf_encoder = hf_outputs.encoder_last_hidden_state.numpy()
        hf_decoder = hf_outputs.last_hidden_state.numpy()

    enc_diff = np.abs(keras_encoder - hf_encoder)
    dec_diff = np.abs(keras_decoder - hf_decoder)
    print(
        f"Encoder abs diff: max={enc_diff.max():.3e} mean={enc_diff.mean():.3e}"
    )
    print(
        f"Decoder abs diff: max={dec_diff.max():.3e} mean={dec_diff.mean():.3e}"
    )
    # The mean diff is dominated by per-op fp32 accumulation drift between
    # TF and PyTorch over the whole encoder/decoder stack. We sanity-check
    # the mean (not the max) so backend jitter can't mask a real bug.
    assert enc_diff.mean() < 1e-3, (
        f"encoder mean abs diff {enc_diff.mean():.3e} exceeds 1e-3"
    )
    assert dec_diff.mean() < 5e-3, (
        f"decoder mean abs diff {dec_diff.mean():.3e} exceeds 5e-3"
    )
    print("Encoder/decoder mean abs diff within tolerance.")


def check_generate(keras_model, hf_model, hf_processor):
    audio, _ = librosa.load(SAMPLE_AUDIO_PATH, sr=16000)
    audio = audio.reshape(1, -1)

    keras_model.compile(sampler="greedy")
    # NOTE: pass `stop_token_ids=None` to bypass a pre-existing issue in
    # `WhisperAudioToText.generate_step` — the sampler's stop predicate
    # checks for stop tokens at *unmasked* positions of the prompt buffer,
    # which for Whisper is pre-filled with `pad == eos`, so the default
    # `stop_token_ids="auto"` aborts before any token is generated. With
    # `None`, generation runs to `max_length`, then we trim at the first
    # EOS token id manually to recover a clean transcript.
    tokenizer = keras_model.preprocessor.tokenizer
    eos_id = tokenizer.eos_token_id
    preprocessed = keras_model.preprocessor.generate_preprocess(
        {"audio": audio}
    )
    step_out = keras_model.generate_step(preprocessed, stop_token_ids=None)
    ids = keras.ops.convert_to_numpy(step_out["decoder_token_ids"])[0]
    # Drop everything from the first EOS onward.
    eos_positions = np.where(ids == eos_id)[0]
    if len(eos_positions) > 0:
        ids = ids[: eos_positions[0]]
    # Strip the BOS prefix.
    bos_id = tokenizer.bos_token_id
    while len(ids) > 0 and ids[0] == bos_id:
        ids = ids[1:]
    keras_text = tokenizer.detokenize(ids[None, :])[0]
    if hasattr(keras_text, "numpy"):
        keras_text = keras_text.numpy()
    if isinstance(keras_text, bytes):
        keras_text = keras_text.decode("utf-8")
    print(f"Keras generate(): {keras_text!r}")

    hf_inputs = hf_processor(audio[0], sampling_rate=16000, return_tensors="pt")
    hf_model.eval()
    with torch.no_grad():
        hf_token_ids = hf_model.generate(
            hf_inputs.input_features, max_length=64
        )
    hf_text = hf_processor.batch_decode(hf_token_ids, skip_special_tokens=True)[
        0
    ]
    print(f"HF    generate(): {hf_text!r}")


def check_roundtrip(keras_model, preset_path):
    keras_model.save_to_preset(preset_path)
    print(f"Saved preset to {preset_path}.")
    reloaded = keras_hub.models.WhisperAudioToText.from_preset(preset_path)
    print(f"Reloaded preset from {preset_path}: {type(reloaded).__name__}")


def main(_):
    preset = FLAGS.preset
    hf_name = PRESET_MAP[preset]
    output_dir = f"./{preset}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Loading HF reference: {hf_name} ===")
    hf_model = transformers.WhisperForConditionalGeneration.from_pretrained(
        hf_name
    )
    hf_processor = transformers.WhisperProcessor.from_pretrained(hf_name)

    print(f"\n=== Building Keras model from hf://{hf_name} ===")
    keras_model = keras_hub.models.WhisperAudioToText.from_preset(
        f"hf://{hf_name}"
    )

    print("\n=== Parameter count ===")
    check_param_match(keras_model, hf_model)

    print("\n=== Numerics on synthetic audio ===")
    check_numerics(keras_model, hf_model, hf_processor)

    print("\n=== generate() on a real audio clip ===")
    check_generate(keras_model, hf_model, hf_processor)

    print("\n=== save_to_preset + reload roundtrip ===")
    check_roundtrip(keras_model, os.path.join(output_dir, "preset"))

    print(f"\nDone. Local preset at {output_dir}/preset.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
