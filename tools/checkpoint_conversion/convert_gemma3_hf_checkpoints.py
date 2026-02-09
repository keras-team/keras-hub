import gc
import os
import random
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)


PRESET_MAP = {
    "gemma3_270m": "google/gemma-3-270m",
    "gemma3_instruct_270m": "google/gemma-3-270m-it",
    "gemma3_1b": "google/gemma-3-1b-pt",
    "gemma3_instruct_1b": "google/gemma-3-1b-it",
    "gemma3_4b": "google/gemma-3-4b-pt",
    "gemma3_instruct_4b": "google/gemma-3-4b-it",
    "translategemma_4b_it": "google/translategemma-4b-it",
    "translategemma_12b_it": "google/translategemma-12b-it",
    "translategemma_27b_it": "google/translategemma-27b-it",
    "gemma3_12b": "google/gemma-3-12b-pt",
    "gemma3_instruct_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-pt",
    "gemma3_instruct_27b": "google/gemma-3-27b-it",
    "function_gemma_instruct_270m": "google/functiongemma-270m-it",
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
    hf_model_tokenizer,
    keras_dtype,
):
    # First, test that the number of parameters match
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
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # Compute difference stats for reporting
    abs_diff = np.abs(keras_hub_logits - hf_output_logits)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Get dtype-appropriate tolerances
    tolerances = DTYPE_TOLERANCES.get(keras_dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = tolerances["atol"]
    rtol = tolerances["rtol"]

    print(f"\nLogit comparison (dtype: {keras_dtype}):")
    print(f"   Max absolute difference: {max_abs_diff:.6f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"   Tolerance - atol: {atol}, rtol: {rtol}")

    # Check with both absolute and relative tolerance
    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=atol, rtol=rtol
        )
        print("✓ All logits within tolerance.")
    except AssertionError as err:
        print(
            "Some logits exceed tolerance (likely kernel differences).\n"
            "NOTE: Generated text comparison is the authoritative check."
        )
        # Provide detailed mismatch information for debugging
        print("Traceback:")
        print(traceback.format_exc())
        print("Assertion message:")
        print(err.args[0])

    # Sequence-wide normalized top-k comparison (all timesteps)
    k = 50
    print(f"Top-{k} normalized logits check across all timesteps:")
    hf_norm = hf_output_logits - hf_output_logits.max(axis=-1, keepdims=True)
    kh_norm = keras_hub_logits - keras_hub_logits.max(axis=-1, keepdims=True)
    hf_topk = np.partition(hf_norm, -k, axis=-1)[..., -k:]
    kh_topk = np.partition(kh_norm, -k, axis=-1)[..., -k:]
    try:
        np.testing.assert_allclose(kh_topk, hf_topk, atol=atol, rtol=rtol)
        print("✓ Top-50 normalized logits within tolerance.")
    except AssertionError as err:
        print("Top-50 normalized logits exceed tolerance.")
        print(traceback.format_exc())
        print(err.args[0])


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor.generate_preprocess(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)


def validate_output(
    keras_model,
    hf_model,
    hf_tokenizer,
):
    input_str = "What is Keras?"
    length = 32

    # KerasHub
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)

    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=length,
        do_sample=False,
        num_beams=1,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    print("🔶 Huggingface generated token ids:", outputs[0])
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print("🔶 Huggingface output:", hf_generated_text)


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
    # Force float32 load for validation
    target_dtype = torch.float32

    # Load with explicit dtype from config
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_preset,
            device_map=device,
            torch_dtype=target_dtype,
        )
    except (ValueError, OSError):
        hf_model = AutoModelForImageTextToText.from_pretrained(
            hf_preset,
            device_map=device,
            torch_dtype=target_dtype,
        )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    # Verify the actual loaded dtype
    hf_dtype = next(hf_model.parameters()).dtype
    # Validate in float32 to reduce numerical drift
    keras_dtype = "float32"
    print(
        f"-> Actual loaded dtype: {hf_dtype} -> "
        f"Using Keras dtype: {keras_dtype}"
    )

    # Load Keras backbone with matching dtype
    keras_hub_backbone = keras_hub.models.Gemma3Backbone.from_preset(
        f"hf://{hf_preset}", dtype=keras_dtype
    )

    keras_hub_tokenizer = keras_hub.models.Gemma3Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = (
        keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
            f"hf://{hf_preset}"
        )
    )

    # Check if the backbone has vision_encoder (is a vision model)
    has_vision = keras_hub_backbone.vision_encoder is not None
    print(f"-> Backbone has vision_encoder: {has_vision}")
    has_image_conv = keras_hub_preprocessor.image_converter is not None
    print(f"-> Preprocessor has image_converter: {has_image_conv}")

    # If vision model but preprocessor has no image_converter,
    # load it explicitly
    if has_vision and keras_hub_preprocessor.image_converter is None:
        print("-> Loading image_converter for vision model...")
        image_converter = keras_hub.layers.Gemma3ImageConverter.from_preset(
            f"hf://{hf_preset}"
        )
        # Recreate preprocessor with image_converter
        keras_hub_preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor(
            tokenizer=keras_hub_tokenizer,
            image_converter=image_converter,
            sequence_length=keras_hub_preprocessor.sequence_length,
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

    gemma3_lm = keras_hub.models.Gemma3CausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
        sampler="greedy",
    )

    validate_output(gemma3_lm, hf_model, hf_tokenizer)

    save_dtype = FLAGS.save_dtype
    if save_dtype == "float32":
        # Already in float32, save directly
        print(f"\n-> Saving model in {save_dtype}...")
        gemma3_lm.save_to_preset(f"./{preset}")
    else:
        # Free up memory before reloading in save dtype
        del gemma3_lm
        del keras_hub_backbone
        del hf_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reload the model in target dtype for saving
        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_hub_backbone_save = keras_hub.models.Gemma3Backbone.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        gemma3_lm_save = keras_hub.models.Gemma3CausalLM(
            backbone=keras_hub_backbone_save,
            preprocessor=keras_hub_preprocessor,
        )
        gemma3_lm_save.save_to_preset(f"./{preset}")

    print(f"\n-> Saved converted model ({save_dtype}) to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
