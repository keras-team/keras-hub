"""Checkpoint conversion script for Qwen3.5 (text-only CausalLM).

Usage:
    python tools/checkpoint_conversion/convert_qwen3_5_checkpoints.py \
        --preset qwen3_5_7b_en

This script:
1. Loads the HF model and tokenizer
2. Loads the KerasHub model via from_preset("hf://...")
3. Compares tokenizer outputs
4. Compares model logits (forward pass)
5. Compares generated text (greedy decoding)
6. Saves the KerasHub preset
"""

import os
import random
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import numpy as np
import torch
from absl import app
from absl import flags

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "qwen3_5_0.8b_base": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3_5_0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_2b_base": "Qwen/Qwen3.5-2B-Base",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b_base": "Qwen/Qwen3.5-4B-Base",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_9b_base": "Qwen/Qwen3.5-9B-Base",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    "qwen3_5_27b": "Qwen/Qwen3.5-27B",
    "qwen3_5_35b_a3b_base": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen3_5_35b_a3b": "Qwen/Qwen3.5-35B-A3B",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(
    keras_hub_model,
    keras_hub_tokenizer,
    hf_model,
    hf_model_tokenizer,
):
    """Compare parameter counts and forward pass outputs."""
    # 1. Parameter count.
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    print(f"\n-> KerasHub params: {keras_hub_params:,}")
    print(f"-> HF params:      {hf_params:,}")
    if keras_hub_params != hf_params:
        print(
            f"WARNING: Parameter count mismatch! "
            f"Diff: {abs(keras_hub_params - hf_params):,}"
        )
    else:
        print("-> Parameter counts match!")

    # 2. Forward pass comparison.
    test_text = "What is Keras?"
    hf_inputs = hf_model_tokenizer([test_text], return_tensors="pt").to(device)
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_preprocessor = keras_hub.models.Qwen3_5CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    seq_len = hf_inputs["input_ids"].shape[1]
    keras_hub_inputs = keras_hub_preprocessor(
        [test_text], sequence_length=seq_len
    )[0]

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # Compare.
    mean_diff = np.mean(np.abs(keras_hub_logits - hf_output_logits))
    max_diff = np.max(np.abs(keras_hub_logits - hf_output_logits))
    print(f"\n-> Logits mean absolute diff: {mean_diff:.6f}")
    print(f"-> Logits max absolute diff:  {max_diff:.6f}")

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-2
        )
        print("-> Forward pass outputs match! (atol=1e-2)")
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")

    # Print first few logits for manual inspection.
    print("\n-> KerasHub logits (first 5 of last token):")
    print(f"   {keras_hub_logits[0, -1, :5]}")
    print("-> HF logits (first 5 of last token):")
    print(f"   {hf_output_logits[0, -1, :5]}")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    """Compare tokenizer outputs."""
    test_text = "What is Keras?"
    hf_output = hf_tokenizer([test_text], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()

    keras_hub_preprocessor = keras_hub.models.Qwen3_5CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    seq_len = hf_output.shape[1]
    keras_hub_output = keras_hub_preprocessor(
        [test_text], sequence_length=seq_len
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    print(f"\n-> HF token ids:      {hf_output[0]}")
    print(f"-> KerasHub token ids: {keras_hub_output[0]}")

    try:
        np.testing.assert_equal(keras_hub_output, hf_output)
        print("-> Tokenizer outputs match!")
    except AssertionError as err:
        print(f"WARNING: Tokenizer mismatch: {err}")


def validate_output(keras_model, hf_model, hf_tokenizer):
    """Compare greedy generation outputs."""
    input_str = "What is Keras?"
    length = 32

    print(f"\n-> Generating with max_length={length}...")

    # KerasHub generation.
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print(f"\n   KerasHub output: {keras_output}")

    # HF generation.
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=length,
        do_sample=False,
        num_beams=1,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    print(f"   HF generated token ids: {outputs[0]}")
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print(f"   HF output: {hf_generated_text}")

    # Compare.
    if keras_output.strip() == hf_generated_text.strip():
        print("\n-> Generated text MATCHES!")
    else:
        print(
            "\n-> Generated text DIFFERS (may be expected for "
            "long sequences due to floating point drift)"
        )


def main(_):
    # === Validate preset ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print(f"=== Converting {preset} ({hf_preset}) ===")

    # === Load HF model ===
    print("\n-> Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=torch.float32,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()
    print(f"   HF model loaded: {hf_model.num_parameters():,} params")

    # === Load KerasHub model ===
    print("\n-> Loading KerasHub model from HF preset...")
    keras_hub_backbone = keras_hub.models.Qwen3_5Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.models.Qwen3_5Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("   KerasHub model loaded!")

    # === Run comparisons ===
    print("\n" + "=" * 50)
    print("TOKENIZER COMPARISON")
    print("=" * 50)
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    test_model(
        keras_hub_backbone,
        keras_hub_tokenizer,
        hf_model,
        hf_tokenizer,
    )

    print("\n" + "=" * 50)
    print("GENERATION COMPARISON")
    print("=" * 50)
    preprocessor = keras_hub.models.Qwen3_5CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    qwen3_5_lm = keras_hub.models.Qwen3_5CausalLM(
        backbone=keras_hub_backbone,
        preprocessor=preprocessor,
        sampler="greedy",
    )
    validate_output(qwen3_5_lm, hf_model, hf_tokenizer)

    # === Save preset ===
    output_dir = f"./{preset}"
    print(f"\n-> Saving KerasHub preset to {output_dir}...")
    qwen3_5_lm.save_to_preset(output_dir)
    print(f"   Preset saved to {output_dir}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
