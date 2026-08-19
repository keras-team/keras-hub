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
        max_new_tokens=20,
        do_sample=False,
    )
    # Extract only new tokens for comparison
    hf_new_tokens = outputs[0, hf_inputs.input_ids.shape[-1] :]
    hf_output_text = hf_processor.tokenizer.decode(
        hf_new_tokens, skip_special_tokens=True
    ).strip()
    print("-> Huggingface generation finished.")

    # Free HF memory to avoid OOM/Hang during Keras compilation
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keras input
    print("-> Starting KerasHub generation...")
    keras_inputs = {"audio": audio_data, "prompts": "transcribe"}

    # Run inference
    print("   Running inference...")
    # Get prompt length and stop tokens
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
    keras_new_tokens = keras_outputs["token_ids"][0, prompt_len:]
    keras_output_text = keras_hub_tokenizer.detokenize(
        keras_new_tokens, skip_special_tokens=True
    ).strip()

    keras_hub_model.preprocessor = old_preprocessor
    print("-> KerasHub generation finished.")

    print(f"🔶 KerasHub output:    '{keras_output_text}'")
    print(f"🔶 Huggingface output: '{hf_output_text}'")


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
