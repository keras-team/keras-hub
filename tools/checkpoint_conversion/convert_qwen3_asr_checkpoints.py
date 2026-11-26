"""Convert Qwen3-ASR HuggingFace checkpoints to KerasHub presets and verify."""

import gc
import os
import random
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from absl import app
from absl import flags

random.seed(123)
torch.manual_seed(123)
# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

import keras  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from keras import ops  # noqa: E402
from transformers import AutoModelForSpeechSeq2Seq  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import keras_hub  # noqa: E402

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "Qwen3-ASR checkpoint conversion tests require pyarrow. "
        "Please install it via `pip install pyarrow`."
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:
    raise ImportError(
        "Qwen3-ASR checkpoint conversion tests require soundfile. "
        "Please install it via `pip install soundfile`."
    ) from exc

keras.config.set_dtype_policy("float32")

PRESET_MAP = {
    "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3_asr_1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
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


def test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_processor):
    audio_path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    audio = pq.read_table(audio_path, columns=["audio"]).slice(0, 1)
    audio_bytes = audio.column("audio")[0].as_py()["bytes"]
    audio_data, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")

    messages = [
        {"role": "system", "content": "transcribe"},
        {
            "role": "user",
            "content": [{"type": "audio", "audio_url": "dummy"}],
        },
    ]
    chat_prompt = hf_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    keras_prompt = chat_prompt.replace(audio_placeholder, "<audio>")
    keras_parameter_count = keras_hub_model.count_params()
    hf_parameter_count = sum(
        parameter.numel() for parameter in hf_model.parameters()
    )
    print(
        "-> Parameter counts: "
        f"KerasHub={keras_parameter_count}, HF={hf_parameter_count}"
    )
    if keras_parameter_count != hf_parameter_count:
        raise ValueError("KerasHub and HF parameter counts do not match.")

    audio_token_ids = [
        keras_hub_tokenizer.vocabulary[token]
        for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>")
    ]
    short_audio = audio_data[: int(0.4 * sample_rate)]
    long_audio_samples = int(31 * sample_rate)
    long_audio = np.tile(
        audio_data,
        int(np.ceil(long_audio_samples / len(audio_data))),
    )[:long_audio_samples]
    audio_cases = [
        ("LibriSpeech", [audio_data]),
        ("short (<0.5s)", [short_audio]),
        ("long (>30s)", [long_audio]),
        ("mixed-length batch", [audio_data, short_audio]),
    ]

    original_hf_inputs = None
    original_keras_inputs = None
    original_prompt_len = None
    for case_name, case_audios in audio_cases:
        batch_size = len(case_audios)
        hf_inputs = hf_processor(
            text=[chat_prompt] * batch_size,
            audio=case_audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        sequence_length = hf_inputs.input_ids.shape[-1] + 128
        keras_inputs = {
            "audio": case_audios,
            "prompts": [keras_prompt] * batch_size,
        }
        preprocessed = keras_hub_model.preprocessor.generate_preprocess(
            keras_inputs, sequence_length=sequence_length
        )

        for b in range(batch_size):
            keras_token_ids = ops.convert_to_numpy(preprocessed["token_ids"])[b]
            keras_padding_mask = ops.convert_to_numpy(
                preprocessed["padding_mask"]
            )[b].astype(bool)
            hf_attention_mask = hf_inputs.attention_mask[b].bool()
            hf_token_ids = (
                hf_inputs.input_ids[b][hf_attention_mask].cpu().numpy()
            )
            if not np.array_equal(
                keras_token_ids[keras_padding_mask], hf_token_ids
            ):
                raise ValueError(
                    f"KerasHub and HF prompt token IDs do not match for "
                    f"{case_name} audio (item {b})."
                )

            keras_audio_mask = ops.convert_to_numpy(preprocessed["audio_mask"])[
                b
            ]
            keras_mel_length = int(np.sum(keras_audio_mask))
            hf_mel_length = int(hf_inputs.input_features_mask[b].sum())
            if keras_mel_length != hf_mel_length:
                raise ValueError(
                    f"KerasHub and HF audio mask lengths do not match for "
                    f"{case_name} audio (item {b}): "
                    f"{keras_mel_length} vs {hf_mel_length}."
                )

            keras_mel = ops.convert_to_numpy(preprocessed["audio_mel"])[b]
            hf_mel = hf_inputs.input_features[b].transpose(0, 1).cpu().numpy()
            mel_diff = np.max(
                np.abs(keras_mel[:keras_mel_length] - hf_mel[:hf_mel_length])
            )
            print(
                f"-> {case_name}[{b}] preprocessing mel max abs diff: "
                f"{mel_diff:.6g}"
            )
            if mel_diff > 1e-3:
                raise ValueError(
                    f"KerasHub and HF mel features differ for {case_name}[{b}] "
                    f"audio by {mel_diff:.6g}."
                )

        with torch.no_grad():
            keras_logits_all = keras_hub_model(preprocessed, training=False)
            hf_logits_all = hf_model(**hf_inputs.to(device)).logits

        for b in range(batch_size):
            keras_logits = ops.convert_to_numpy(keras_logits_all)[b]
            keras_token_ids_all = ops.convert_to_numpy(
                preprocessed["token_ids"]
            )[b]
            keras_valid_text_positions = (
                ops.convert_to_numpy(preprocessed["padding_mask"])[b].astype(
                    bool
                )
            ) & ~np.isin(keras_token_ids_all, audio_token_ids)
            if bool(hf_inputs.attention_mask[b].all()):
                hf_logits = hf_logits_all[b].detach().cpu().numpy()
                hf_token_ids_all = hf_inputs.input_ids[b].cpu().numpy()
                hf_valid_text_positions = (
                    hf_inputs.attention_mask[b].cpu().numpy().astype(bool)
                ) & ~np.isin(hf_token_ids_all, audio_token_ids)
            else:
                # HF Qwen3ASRForConditionalGeneration.forward does not preserve
                # single-clip logits on padded batch rows, whereas KerasHub's
                # right-padded forward is batch-invariant (~3.6e-5).
                hf_single = hf_processor(
                    text=[chat_prompt],
                    audio=[case_audios[b]],
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    hf_logits = (
                        hf_model(**hf_single.to(device))
                        .logits[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                hf_token_ids_all = hf_single.input_ids[0].cpu().numpy()
                hf_valid_text_positions = (
                    hf_single.attention_mask[0].cpu().numpy().astype(bool)
                ) & ~np.isin(hf_token_ids_all, audio_token_ids)
            diffs = np.abs(
                keras_logits[keras_valid_text_positions]
                - hf_logits[hf_valid_text_positions]
            )
            logits_max_diff = np.max(diffs)
            logits_mean_diff = np.mean(diffs)
            print(
                f"-> {case_name}[{b}] logits diff: "
                f"max={logits_max_diff:.6g}, mean={logits_mean_diff:.6g}"
            )
            if logits_max_diff > 1e-3:
                raise ValueError(
                    f"KerasHub and HF logits differ for {case_name}[{b}] "
                    f"audio by {logits_max_diff:.6g}."
                )

        if case_name == "LibriSpeech":
            original_hf_inputs = hf_inputs
            original_keras_inputs = {
                "audio": audio_data,
                "prompts": keras_prompt,
            }
            keras_padding_mask = ops.convert_to_numpy(
                preprocessed["padding_mask"]
            )[0].astype(bool)
            original_prompt_len = int(np.sum(keras_padding_mask))

    keras_output_text = keras_hub_model.generate(
        original_keras_inputs,
        max_length=original_prompt_len + 20,
        strip_prompt=True,
    )
    keras_output_text = str(keras_output_text).strip()

    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids=original_hf_inputs.input_ids.to(device),
            input_features=original_hf_inputs.input_features.to(device),
            input_features_mask=original_hf_inputs.input_features_mask.to(
                device
            ),
            max_new_tokens=20,
            do_sample=False,
        )
    hf_new_tokens = hf_outputs[0, original_hf_inputs.input_ids.shape[-1] :]
    hf_output_text = hf_processor.tokenizer.decode(
        hf_new_tokens, skip_special_tokens=True
    ).strip()
    print(f"-> KerasHub output:    '{keras_output_text}'")
    print(f"-> Huggingface output: '{hf_output_text}'")

    if keras_output_text != hf_output_text:
        raise ValueError(
            "KerasHub and Huggingface outputs do not match! "
            f"KerasHub: {keras_output_text}, HF: {hf_output_text}"
        )

    # Free HF memory after comparison
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    save_dtype = FLAGS.save_dtype

    # === Load the Huggingface model ===
    print(f"Loading HF model {hf_preset}...")
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_preset,
        device_map=device,
        dtype=torch.float32,
    )
    hf_processor = AutoProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    print(f"Loading Keras Hub model hf://{hf_preset}...")
    keras_hub_model = keras_hub.models.Qwen3ASRCausalLM.from_preset(
        f"hf://{hf_preset}"
    )

    print("\n-> Checking outputs...")
    test_model(
        keras_hub_model,
        keras_hub_model.preprocessor.tokenizer,
        hf_model,
        hf_processor,
    )
    print("\n-> Tests passed!")

    if save_dtype != "float32":
        del keras_hub_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Reloading Keras Hub model with dtype={save_dtype}...")
        keras_hub_model = keras_hub.models.Qwen3ASRCausalLM.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )

    print(f"Saving to preset ./{preset} ({save_dtype})...")
    keras_hub_model.save_to_preset(f"./{preset}")
    print("-> Preset saved successfully!")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
