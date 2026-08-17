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
    # Use 30 seconds to match model default max_audio_length
    audio_data = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate * 30) / sample_rate
    ).astype("float32")

    # Keras input
    print("-> Running KerasHub generation...")
    # The KerasHub preprocessor will prepend audio tokens if <audio> is missing.
    keras_inputs = {"audio": audio_data, "prompts": "transcribe"}
    # Use max_length slightly above prompt length (~395) for speed
    old_preprocessor = keras_hub_model.preprocessor
    keras_hub_model.preprocessor = None
    preprocessed = old_preprocessor(keras_inputs)
    # Add batch dimension
    preprocessed = {
        k: ops.expand_dims(v, 0)
        if ops.ndim(v) < (3 if k == "audio_mel" else 2)
        else v
        for k, v in preprocessed.items()
    }
    keras_output = keras_hub_model.generate(
        preprocessed, max_length=400, stop_token_ids=None
    )
    decoded_keras = keras_hub_tokenizer.detokenize(
        keras_output["token_ids"], skip_special_tokens=True
    )
    keras_hub_model.preprocessor = old_preprocessor
    print("-> KerasHub generation finished.")

    # HF input
    print("-> Running Huggingface generation...")
    hf_prompt = "<|audio_start|><|audio_pad|><|audio_end|>\ntranscribe"
    hf_inputs = hf_processor(
        text=hf_prompt,
        audio=audio_data,
        sampling_rate=16000,
        return_tensors="pt",
    ).to(device)

    # HF generate
    outputs = hf_model.generate(
        **hf_inputs,
        max_new_tokens=5,
        do_sample=False,
    )
    hf_output_text = hf_processor.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print("-> Huggingface generation finished.")

    # Clean up both for comparison
    if isinstance(decoded_keras, list):
        decoded_keras = decoded_keras[0]
    keras_final = decoded_keras.replace("transcribe", "").strip()
    hf_final = hf_output_text.replace("transcribe", "").strip()

    print("🔶 KerasHub output:", keras_final)
    print("🔶 Huggingface output:", hf_final)


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
