import os
import random

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
from keras import ops  # noqa: E402
from transformers import AutoModelForSpeechSeq2Seq  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import keras_hub  # noqa: E402

keras.config.set_dtype_policy("float32")

PRESET_MAP = {
    "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_processor):
    # Test with dummy/synthetic audio
    sample_rate = 16000
    # Use 5 seconds for faster testing
    audio_data = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate * 5) / sample_rate
    ).astype("float32")

    # Keras input
    print("-> Starting KerasHub generation...")
    keras_inputs = {"audio": audio_data, "prompts": "transcribe"}

    # Run inference
    print("   Running inference...")
    # Get prompt tokens for HF parity
    preprocessed = keras_hub_model.preprocessor.generate_preprocess(
        keras_inputs, sequence_length=256
    )
    prompt_len = int(ops.sum(ops.cast(preprocessed["padding_mask"], "int32")))

    stop_ids = [keras_hub_tokenizer.end_token_id]
    if hasattr(keras_hub_tokenizer, "end_token2_id"):
        stop_ids.append(keras_hub_tokenizer.end_token2_id)

    # Detach preprocessor to get raw token IDs
    old_preprocessor = keras_hub_model.preprocessor
    keras_hub_model.preprocessor = None

    # Prepare batch
    batch = {
        k: ops.expand_dims(ops.convert_to_tensor(v), 0)
        for k, v in preprocessed.items()
    }

    keras_outputs = keras_hub_model.generate(
        batch, max_length=prompt_len + 20, stop_token_ids=stop_ids
    )

    # Extract only new tokens and detokenize skipping special ones
    # Truncate to 20 tokens to match HF max_new_tokens=20
    keras_new_tokens = keras_outputs["token_ids"][
        0, prompt_len : prompt_len + 20
    ]
    keras_output_text = keras_hub_tokenizer.detokenize(
        keras_new_tokens, skip_special_tokens=True
    )
    if isinstance(keras_output_text, torch.Tensor):
        keras_output_text = keras_output_text.cpu().numpy()
    if isinstance(keras_output_text, np.ndarray):
        keras_output_text = keras_output_text.item()
    if isinstance(keras_output_text, bytes):
        keras_output_text = keras_output_text.decode("utf-8")
    keras_output_text = keras_output_text.strip()

    keras_hub_model.preprocessor = old_preprocessor
    print("-> KerasHub generation finished.")

    # HF input
    print("-> Running Huggingface generation...")
    # Use the EXACT same token IDs as KerasHub to ensure prompt parity
    hf_input_ids = torch.from_numpy(
        ops.convert_to_numpy(batch["token_ids"])
    ).to(device)
    hf_input_ids = hf_input_ids[:, :prompt_len]  # Only the valid prompt part

    hf_inputs = hf_processor(
        text="dummy",
        audio=audio_data,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
    )
    hf_audio_mel = hf_inputs.input_features.to(device)
    hf_audio_mask = hf_inputs.input_features_mask.to(device)

    # HF generate
    outputs = hf_model.generate(
        input_ids=hf_input_ids,
        input_features=hf_audio_mel,
        input_features_mask=hf_audio_mask,
        max_new_tokens=20,
        do_sample=False,
    )
    # Extract only new tokens for comparison
    hf_new_tokens = outputs[
        0, hf_input_ids.shape[-1] : hf_input_ids.shape[-1] + 20
    ]
    hf_output_text = hf_processor.tokenizer.decode(
        hf_new_tokens, skip_special_tokens=True
    ).strip()
    print("-> Huggingface generation finished.")

    print(f"🔶 KerasHub output:    '{keras_output_text}'")
    print(f"🔶 Huggingface output: '{hf_output_text}'")

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
