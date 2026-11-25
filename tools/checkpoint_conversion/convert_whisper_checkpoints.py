"""Convert Whisper checkpoints from Hugging Face to KerasHub format.

The on-the-fly converter at `keras_hub/src/utils/transformers/convert_whisper`
does the actual weight porting. This script runs it end to end for a single
preset and checks the result against the Hugging Face reference:

1. Load the HF reference model and processor.
2. Build the Keras task model with `from_preset("hf://...")`.
3. Assert that both models have the same number of parameters.
4. Compare encoder and decoder outputs on synthetic audio, given the same
   HF log-mel features.
5. Compare multi-token decoder logits end to end, from raw audio through
   each model's own frontend, on a clip shorter than 0.5 seconds, a clip
   longer than 30 seconds, and a batch of both.
6. Run greedy `generate()` on a real clip with both models and assert that
   the transcripts match.
7. Save the Keras preset, reload it, and assert that the reloaded model
   gives the same output.

Usage:
```shell
python tools/checkpoint_conversion/convert_whisper_checkpoints.py \
    --preset whisper_tiny_en
```
"""

import copy
import os

import keras
import numpy as np
import torch
import transformers
from absl import app
from absl import flags

import keras_hub

try:
    import librosa
except ImportError:
    raise ImportError(
        "Whisper checkpoint conversion requires librosa to load the sample "
        "audio. Please install it with `pip install librosa`."
    )

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
    "whisper_large_multi_v3": "openai/whisper-large-v3",
    "whisper_large_multi_v3_turbo": "openai/whisper-large-v3-turbo",
}

# Resolve the sample clips from this file, so the script runs from any
# working directory.
TEST_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "keras_hub/src/tests/test_data/audio_transcription_tests",
    )
)
SHORT_AUDIO_PATH = os.path.join(TEST_DATA_DIR, "male_short_voice_clip_3sec.wav")
LONG_AUDIO_PATH = os.path.join(
    TEST_DATA_DIR, "male_muffled_voice_clip_46sec.wav"
)
SAMPLING_RATE = 16000

# fp32 drift between the PyTorch and Keras stacks grows with model depth;
# these bounds catch a wrong weight or a wrong mask, not backend jitter.
HIDDEN_STATE_ATOL = 5e-3
LOGITS_ATOL = 5e-2
LOGITS_RTOL = 1e-3

# A forced transcript prefix gives several decoder steps to compare.
TEXT_PREFIX = " The quick brown fox"
GENERATE_MAX_LENGTH = 64

FLAGS = flags.FLAGS
flags.DEFINE_enum("preset", None, list(PRESET_MAP), "The preset to convert.")


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLING_RATE)
    return audio.astype("float32")


def check_param_match(keras_model, hf_model):
    keras_params = keras_model.backbone.count_params()
    hf_params = sum(p.numel() for p in hf_model.model.parameters())
    print(f"Keras backbone params: {keras_params:,}")
    print(f"HF      model params: {hf_params:,}")
    assert keras_params == hf_params, (
        f"Parameter counts differ: Keras {keras_params:,}, HF {hf_params:,}"
    )


def check_hidden_states(keras_model, hf_model, hf_processor):
    """Compare encoder/decoder outputs given identical log-mel features.

    Both stacks get the same HF-extracted features, so this isolates the
    encoder and decoder weights from the audio frontend.
    """
    rng = np.random.default_rng(0)
    raw_audio = rng.standard_normal((SAMPLING_RATE,)).astype("float32")
    hf_features = hf_processor.feature_extractor(
        raw_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features  # (1, n_mels, n_frames)

    decoder_token_ids = np.array(
        [keras_model.preprocessor.decoder_prefix_ids], dtype="int32"
    )
    keras_outputs = keras_model.backbone(
        {
            # The Keras backbone takes (batch, n_frames, n_mels).
            "encoder_features": np.transpose(hf_features.numpy(), (0, 2, 1)),
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": np.ones_like(decoder_token_ids, "bool"),
        },
        training=False,
    )
    with torch.no_grad():
        hf_outputs = hf_model.model(
            input_features=hf_features,
            decoder_input_ids=torch.from_numpy(decoder_token_ids).long(),
        )

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(keras_outputs["encoder_sequence_output"]),
        hf_outputs.encoder_last_hidden_state.numpy(),
        atol=HIDDEN_STATE_ATOL,
        err_msg="Encoder outputs differ.",
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(keras_outputs["decoder_sequence_output"]),
        hf_outputs.last_hidden_state.numpy(),
        atol=HIDDEN_STATE_ATOL,
        err_msg="Decoder outputs differ.",
    )
    print("Encoder and decoder outputs match.")


def check_logits(keras_model, hf_model, hf_processor):
    """Compare logits end to end, from raw audio, for several decoder steps.

    Each model runs its own frontend, so this also checks
    `WhisperAudioConverter` against `WhisperFeatureExtractor`, including
    padding a short clip and truncating a clip longer than 30 seconds.
    """
    short_audio = load_audio(SHORT_AUDIO_PATH)[: SAMPLING_RATE // 4]
    long_audio = load_audio(LONG_AUDIO_PATH)
    assert len(short_audio) < SAMPLING_RATE // 2
    assert len(long_audio) > 30 * SAMPLING_RATE
    text = TEXT_PREFIX
    preprocessor = keras_model.preprocessor
    num_text_tokens = len(preprocessor.tokenizer(text))
    sequence_length = len(preprocessor.decoder_prefix_ids) + num_text_tokens

    cases = {
        "short clip": [short_audio],
        "long clip": [long_audio],
        "mixed batch": [short_audio, long_audio],
    }
    for name, clips in cases.items():
        # Pad to a common length for Keras; the converter pads to 30 seconds
        # anyway, so trailing zeros do not change the features.
        max_len = max(len(clip) for clip in clips)
        audio = np.stack(
            [np.pad(clip, (0, max_len - len(clip))) for clip in clips]
        )
        inputs = preprocessor.generate_preprocess(
            {"audio": audio, "text": [text] * len(clips)},
            sequence_length=sequence_length,
        )
        keras_logits = keras.ops.convert_to_numpy(keras_model(inputs))

        hf_features = hf_processor.feature_extractor(
            clips, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features
        decoder_token_ids = keras.ops.convert_to_numpy(
            inputs["decoder_token_ids"]
        )
        with torch.no_grad():
            hf_logits = hf_model(
                input_features=hf_features,
                decoder_input_ids=torch.from_numpy(decoder_token_ids).long(),
            ).logits.numpy()

        diff = np.abs(keras_logits - hf_logits)
        print(f"{name}: logits max abs diff {diff.max():.3e}")
        np.testing.assert_allclose(
            keras_logits,
            hf_logits,
            atol=LOGITS_ATOL,
            rtol=LOGITS_RTOL,
            err_msg=f"Logits differ on the {name}.",
        )
        np.testing.assert_array_equal(
            keras_logits.argmax(-1),
            hf_logits.argmax(-1),
            err_msg=f"Greedy tokens differ on the {name}.",
        )
    print("End-to-end logits match.")


def hf_generate(hf_model, hf_processor, audio, max_length):
    # `transformers` v5 rejects `suppress_tokens` set on both the model
    # config and the generation config, so clear the model config copies and
    # pass one explicit generation config.
    hf_model.config.suppress_tokens = None
    hf_model.config.begin_suppress_tokens = None
    generation_config = copy.deepcopy(hf_model.generation_config)
    generation_config.max_length = max_length
    kwargs = {}
    if getattr(generation_config, "is_multilingual", False):
        # The Keras preprocessor defaults to English transcription and does
        # not detect the language.
        kwargs = {"language": "en", "task": "transcribe"}
    features = hf_processor(
        audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        token_ids = hf_model.generate(
            features, generation_config=generation_config, **kwargs
        )
    return hf_processor.batch_decode(token_ids, skip_special_tokens=True)[0]


def check_generate(keras_model, hf_model, hf_processor):
    audio = load_audio(SHORT_AUDIO_PATH)
    keras_text = keras_model.generate(
        {"audio": audio[None]}, max_length=GENERATE_MAX_LENGTH
    )[0]
    hf_text = hf_generate(
        hf_model, hf_processor, audio, max_length=GENERATE_MAX_LENGTH
    )
    print(f"Keras generate(): {keras_text!r}")
    print(f"HF    generate(): {hf_text!r}")
    assert keras_text.strip(), "Keras generated an empty transcript."
    assert keras_text.strip() == hf_text.strip(), "Transcripts differ."
    return keras_text


def check_roundtrip(keras_model, preset_path, keras_text):
    keras_model.save_to_preset(preset_path)
    print(f"Saved preset to {preset_path}.")
    reloaded = keras_hub.models.WhisperAudioToText.from_preset(preset_path)
    audio = load_audio(SHORT_AUDIO_PATH)
    reloaded_text = reloaded.generate(
        {"audio": audio[None]}, max_length=GENERATE_MAX_LENGTH
    )[0]
    assert reloaded_text == keras_text, (
        f"Reloaded preset transcribes {reloaded_text!r}, "
        f"expected {keras_text!r}."
    )
    print("Reloaded preset gives the same transcript.")


def main(_):
    preset = FLAGS.preset
    hf_name = PRESET_MAP[preset]
    output_dir = f"./{preset}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Loading HF reference: {hf_name} ===")
    hf_model = transformers.WhisperForConditionalGeneration.from_pretrained(
        hf_name
    ).eval()
    hf_processor = transformers.WhisperProcessor.from_pretrained(hf_name)

    print(f"\n=== Building Keras model from hf://{hf_name} ===")
    keras_model = keras_hub.models.WhisperAudioToText.from_preset(
        f"hf://{hf_name}"
    )

    print("\n=== Parameter count ===")
    check_param_match(keras_model, hf_model)

    print("\n=== Hidden states on synthetic audio ===")
    check_hidden_states(keras_model, hf_model, hf_processor)

    print("\n=== End-to-end logits on real audio ===")
    check_logits(keras_model, hf_model, hf_processor)

    print("\n=== generate() on a real audio clip ===")
    keras_text = check_generate(keras_model, hf_model, hf_processor)

    print("\n=== save_to_preset + reload roundtrip ===")
    check_roundtrip(keras_model, os.path.join(output_dir, "preset"), keras_text)

    print(f"\nDone. Local preset at {output_dir}/preset.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
