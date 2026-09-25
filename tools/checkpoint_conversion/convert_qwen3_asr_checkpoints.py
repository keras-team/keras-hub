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
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from keras import ops  # noqa: E402
from transformers import AutoModelForSpeechSeq2Seq  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import keras_hub  # noqa: E402

keras.config.set_dtype_policy("float32")

PRESET_MAP = {
    "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3_asr_1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
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
    hf_inputs = hf_processor(
        text=chat_prompt,
        audio=audio_data,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="longest",
    )

    audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    keras_prompt = chat_prompt.replace(audio_placeholder, "<audio>")
    keras_inputs = {"audio": audio_data, "prompts": keras_prompt}
    preprocessed = keras_hub_model.preprocessor.generate_preprocess(
        keras_inputs, sequence_length=256
    )
    keras_token_ids = ops.convert_to_numpy(preprocessed["token_ids"])
    prompt_len = int(ops.sum(ops.cast(preprocessed["padding_mask"], "int32")))
    hf_attention_mask = hf_inputs.attention_mask[0].bool()
    hf_token_ids = hf_inputs.input_ids[0][hf_attention_mask].cpu().numpy()

    if not np.array_equal(keras_token_ids[:prompt_len], hf_token_ids):
        raise ValueError("KerasHub and HF prompt token IDs do not match.")

    keras_mel = ops.convert_to_numpy(preprocessed["audio_mel"])
    hf_mel = hf_inputs.input_features[0].transpose(0, 1).cpu().numpy()
    hf_mel_length = int(hf_inputs.input_features_mask[0].sum())
    mel_diff = np.max(np.abs(keras_mel - hf_mel[:hf_mel_length]))
    print(f"-> Preprocessing mel max abs diff: {mel_diff:.6g}")
    if mel_diff > 1e-3:
        raise ValueError(
            f"KerasHub and HF mel features differ by {mel_diff:.6g}."
        )

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

    hf_batch = {
        "token_ids": hf_inputs.input_ids.to(device),
        "padding_mask": hf_inputs.attention_mask.to(device),
        "audio_mel": hf_inputs.input_features.transpose(1, 2).to(device),
        "audio_mask": hf_inputs.input_features_mask.to(device),
    }
    old_preprocessor = keras_hub_model.preprocessor
    keras_hub_model.preprocessor = None
    keras_logits = keras_hub_model(hf_batch, training=False)
    keras_hub_model.preprocessor = old_preprocessor
    hf_logits = hf_model(**hf_inputs.to(device)).logits
    keras_logits = ops.convert_to_numpy(keras_logits)
    hf_logits = hf_logits.detach().cpu().numpy()
    audio_token_ids = [
        keras_hub_tokenizer.vocabulary[token]
        for token in ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>")
    ]
    text_positions = ~np.isin(
        hf_inputs.input_ids[0].cpu().numpy(), audio_token_ids
    )
    logits_diff = np.max(
        np.abs(keras_logits[:, text_positions] - hf_logits[:, text_positions])
    )
    print(f"-> Logits max abs diff: {logits_diff:.6g}")
    if logits_diff > 1e-3:
        raise ValueError(f"KerasHub and HF logits differ by {logits_diff:.6g}.")

    stop_ids = [keras_hub_tokenizer.end_token_id]
    if hasattr(keras_hub_tokenizer, "end_token2_id"):
        stop_ids.append(keras_hub_tokenizer.end_token2_id)
    keras_output_text = keras_hub_model.generate(
        keras_inputs,
        max_length=prompt_len + 20,
        stop_token_ids=stop_ids,
    )
    keras_output_text = str(keras_output_text)
    keras_output_text = keras_output_text.rsplit("<|im_start|>assistant\n", 1)[
        -1
    ].strip()

    hf_outputs = hf_model.generate(
        input_ids=hf_inputs.input_ids.to(device),
        input_features=hf_inputs.input_features.to(device),
        input_features_mask=hf_inputs.input_features_mask.to(device),
        max_new_tokens=20,
        do_sample=False,
    )
    hf_new_tokens = hf_outputs[0, hf_inputs.input_ids.shape[-1] :]
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

    # === Load the Huggingface model ===
    print(f"Loading HF model {hf_preset}...")
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_processor = AutoProcessor.from_pretrained(
        hf_preset,
        trust_remote_code=True,
    )
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

    print(f"Saving to preset ./{preset}...")
    keras_hub_model.save_to_preset(f"./{preset}")
    print("-> Preset saved successfully!")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
