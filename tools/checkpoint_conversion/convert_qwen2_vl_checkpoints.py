"""Convert Qwen2-VL checkpoints from HuggingFace to KerasHub format.

Usage:
    python tools/checkpoint_conversion/convert_qwen2_vl_checkpoints.py \
        --preset qwen2_vl_2b_instruct
"""

import gc
import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer

import keras_hub

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)


PRESET_MAP = {
    "qwen2_vl_2b_instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_7b_instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_72b_instruct": "Qwen/Qwen2-VL-72B-Instruct",
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


# Tolerance for logit comparison (float32 only validation).
DTYPE_TOLERANCES = {
    "float32": {"atol": 1e-4, "rtol": 1e-4},
}


def test_model(
    keras_hub_model,
    keras_hub_preprocessor,
    hf_model,
    hf_tokenizer,
    keras_dtype,
):
    # First, test that the number of parameters match.
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params
    print(f"\n✓ Parameter count match: {keras_hub_params:,} params")

    # Test the outputs of both the models using identical inputs.
    keras_hub_inputs = keras_hub_preprocessor.generate_preprocess(
        ["What is Keras?"], sequence_length=6
    )
    hf_inputs = {
        "input_ids": torch.tensor(keras_hub_inputs["token_ids"]).to(device),
        "attention_mask": torch.tensor(keras_hub_inputs["padding_mask"]).to(
            device
        ),
    }
    hf_outputs = hf_model.model(**hf_inputs)
    hf_hidden_states = (
        hf_outputs.last_hidden_state.detach().cpu().float().numpy()
    )

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_hidden = ops.convert_to_numpy(keras_hub_output)

    # Compute difference stats for reporting.
    abs_diff = np.abs(keras_hub_hidden - hf_hidden_states)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Get dtype-appropriate tolerances.
    tolerances = DTYPE_TOLERANCES.get(keras_dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = tolerances["atol"]
    rtol = tolerances["rtol"]

    print(f"\nHidden state comparison (dtype: {keras_dtype}):")
    print(f"   Max absolute difference: {max_abs_diff:.6f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"   Tolerance - atol: {atol}, rtol: {rtol}")

    try:
        np.testing.assert_allclose(
            keras_hub_hidden, hf_hidden_states, atol=atol, rtol=rtol
        )
        print("✓ All hidden states within tolerance.")
    except AssertionError as err:
        print(
            "Some hidden states exceed tolerance.\n"
            "NOTE: Generated text comparison is the authoritative check."
        )
        print("Traceback:")
        print(traceback.format_exc())
        print("Assertion message:")
        print(err.args[0])


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor.generate_preprocess(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output["token_ids"])

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
    # Force float32 load for validation.
    target_dtype = torch.float32

    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=target_dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    # Verify the actual loaded dtype.
    hf_dtype = next(hf_model.parameters()).dtype
    keras_dtype = "float32"
    print(
        f"-> Actual loaded dtype: {hf_dtype} -> "
        f"Using Keras dtype: {keras_dtype}"
    )

    # Load Keras backbone with matching dtype.
    keras_hub_backbone = keras_hub.models.Qwen2VLBackbone.from_preset(
        f"hf://{hf_preset}", dtype=keras_dtype
    )
    keras_hub_tokenizer = keras_hub.models.Qwen2VLTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        keras_hub_tokenizer
    )

    print("\n-> Huggingface model and tokenizer loaded")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(
        keras_hub_backbone,
        keras_hub_preprocessor,
        hf_model,
        hf_tokenizer,
        keras_dtype,
    )
    print("\n-> Tests passed!")

    # === Save the model ===
    keras_hub_lm = keras_hub.models.Qwen2VLCausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
    )

    save_dtype = FLAGS.save_dtype
    if save_dtype == "float32":
        print(f"\n-> Saving model in {save_dtype}...")
        keras_hub_lm.save_to_preset(f"./{preset}")
    else:
        del keras_hub_lm
        del keras_hub_backbone
        del hf_model
        gc.collect()

        # Reload in target dtype for saving.
        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_hub_backbone_save = keras_hub.models.Qwen2VLBackbone.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        keras_hub_lm_save = keras_hub.models.Qwen2VLCausalLM(
            backbone=keras_hub_backbone_save,
            preprocessor=keras_hub_preprocessor,
        )
        keras_hub_lm_save.save_to_preset(f"./{preset}")

    print(f"\n-> Saved converted model ({save_dtype}) to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
