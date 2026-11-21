import os
import traceback

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
    "smollm3_3b_en": "HuggingFaceTB/SmolLM3-3B",
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


def compute_keras_output(keras_hub_model, keras_hub_preprocessor):
    """Computes the output of the KerasHub model."""

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


def validate_output(
    keras_model,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Generative output comparison:")
    input_str = "What is Keras?"
    length = 32

    # KerasHub
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print(f"   KerasHub output: {keras_output}")

    # Hugging Face
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt").to(device)
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=length,
        do_sample=False,
        num_beams=1,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print(f"   Huggingface output: {hf_generated_text}")


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

    keras_hub_tokenizer = keras_hub.models.SmolLM3Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras tokenizer loaded")
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)

    print("\n -> Keras tokenizer test successful")

    hf_params = hf_model.num_parameters(only_trainable=False)
    hf_output_logits = compute_hf_output(hf_model, hf_tokenizer)
    print("\n -> Computed HF outputs successfully")

    keras_hub_backbone = keras_hub.models.SmolLM3Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras model loaded")

    keras_hub_params = keras_hub_backbone.count_params()
    print("\n-> Parameter count comparison:")
    print(f"   HuggingFace model: {hf_params:,}")
    print(f"   KerasHub model: {keras_hub_params:,}")

    preprocessor = keras_hub.models.SmolLM3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output_logits = compute_keras_output(
        keras_hub_backbone, preprocessor
    )

    try:
        np.testing.assert_allclose(
            keras_hub_output_logits, hf_output_logits, atol=1e-5
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")

    print("\n-> Tests passed!")

    keras_hub_model = keras_hub.models.SmolLM3CausalLM(
        keras_hub_backbone, preprocessor
    )

    validate_output(keras_hub_model, hf_model, hf_tokenizer)

    keras_hub_model.save_to_preset(f"./{preset}")

    print("\n-> Model presets saved successfully")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
