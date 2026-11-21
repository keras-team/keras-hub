"""
A conversion script for gpt_oss checkpoints.

This script downloads a gpt_oss model from the Hugging Face hub,
converts it to the Keras format, and saves it as a Keras preset.

Usage:
python convert_gpt_oss_checkpoints.py --preset=gpt_oss_8x7b_en
"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import keras_hub

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

PRESET_MAP = {
    "gpt_oss_20b_en": "openai/gpt-oss-20b",
    "gpt_oss_120b_en": "openai/gpt-oss-120b",
    "gpt_oss_safeguard_20b_en": "openai/gpt-oss-safeguard-20b",
    "gpt_oss_safeguard_120b_en": "openai/gpt-oss-safeguard-120b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def compute_hf_output(hf_model, hf_model_tokenizer):
    """Computes the output of the Hugging Face model."""
    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt").to(
        device
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    return hf_output_logits


def compute_keras_output(keras_hub_model, keras_hub_tokenizer):
    """Computes the output of the KerasHub model."""
    keras_hub_preprocessor = keras_hub.models.GptOssCausalLMPreprocessor(
        keras_hub_tokenizer, add_start_token=False
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )[0]
    keras_hub_inputs = {k: v.to(device) for k, v in keras_hub_inputs.items()}

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_output_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_output_logits = ops.convert_to_numpy(keras_hub_output_logits)
    return keras_hub_output_logits


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    """Tests that the tokenizers are the same."""
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()

    # Use tokenizer directly to avoid preprocessor padding
    keras_hub_output = keras_hub_tokenizer(["What is Keras?"])
    keras_hub_output = ops.convert_to_numpy(keras_hub_output)

    np.testing.assert_equal(keras_hub_output, hf_output)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load the Huggingface model ===
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset, device_map=device, torch_dtype=torch.float32
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()
    print("\n-> Huggingface model and tokenizer loaded")

    keras_hub_tokenizer = keras_hub.models.GptOssTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras tokenizer loaded")
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)

    print("\n -> Keras tokenizer test successful")

    hf_params = hf_model.num_parameters()
    hf_output_logits = compute_hf_output(hf_model, hf_tokenizer)

    generate_length = 32
    hf_inputs = hf_tokenizer(["What is Keras?"], return_tensors="pt").to(device)
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=generate_length,
        pad_token_id=hf_tokenizer.pad_token_id,
        do_sample=False,
    )
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print("\n -> Computed HF outputs successfully")

    del hf_model, hf_tokenizer
    keras_hub_backbone = keras_hub.models.GptOssBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras model loaded")

    keras_hub_params = keras_hub_backbone.count_params()
    print("\n-> Parameter count comparison:")
    print(f"   HuggingFace model: {hf_params:,}")
    print(f"   KerasHub model: {keras_hub_params:,}")
    assert keras_hub_params == hf_params

    keras_hub_output_logits = compute_keras_output(
        keras_hub_backbone, keras_hub_tokenizer
    )

    np.testing.assert_allclose(
        keras_hub_output_logits, hf_output_logits, atol=1e-4
    )

    print("\n-> Tests passed!")

    preprocessor = keras_hub.models.GptOssCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_model = keras_hub.models.GptOssCausalLM(
        keras_hub_backbone, preprocessor
    )

    keras_hub_model.compile(sampler="greedy")
    keras_output = keras_hub_model.generate(
        ["What is Keras?"], max_length=generate_length
    )
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)
    print("🔶 Huggingface output:", hf_generated_text)

    keras_hub_model.save_to_preset(f"./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
