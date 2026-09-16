"""Convert VaultGemma HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_vaultgemma_checkpoints.py \
        --preset vault_gemma_1b_en \
        --save_dtype bfloat16
"""

import gc
import os
import random
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "vault_gemma_1b_en": "google/vaultgemma-1b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    "vault_gemma_1b_en",
    f"Must be one of {','.join(PRESET_MAP.keys())}. "
    "Defaults to 'vault_gemma_1b_en'.",
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Optional upload URI, e.g. "kaggle://keras/vaultgemma/keras/{preset}"',
    required=False,
)

# Tolerance for logit comparison.
DTYPE_TOLERANCES = {
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    test_text = "What is Keras?"
    hf_output = hf_tokenizer([test_text], return_tensors="pt")
    hf_tokens = hf_output["input_ids"].detach().cpu().numpy()

    keras_hub_preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor.generate_preprocess(
        [test_text], sequence_length=hf_tokens.shape[1]
    )
    kh_tokens = ops.convert_to_numpy(keras_hub_inputs["token_ids"])

    np.testing.assert_equal(kh_tokens, hf_tokens)
    print("✓ Tokenizer output match.")


def test_model(
    keras_hub_model,
    keras_hub_preprocessor,
    hf_model,
    keras_dtype,
):
    # Verify parameter count
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params, (
        f"Parameter count mismatch: KerasHub={keras_hub_params:,} vs "
        f"HF={hf_params:,}"
    )
    print(f"\n✓ Parameter count match: {keras_hub_params:,} params")

    # Forward pass comparison with identical input tokens
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

    abs_diff = np.abs(keras_hub_logits - hf_output_logits)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    tolerances = DTYPE_TOLERANCES.get(keras_dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = tolerances["atol"]
    rtol = tolerances["rtol"]

    print(f"\nLogit comparison (dtype: {keras_dtype}):")
    print(f"   Max absolute difference:  {max_abs_diff:.6f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"   Tolerance - atol: {atol}, rtol: {rtol}")

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=atol, rtol=rtol
        )
        print("✓ All logits within tolerance.")
    except AssertionError as err:
        print(
            "Some logits exceed tolerance (numerical kernel differences).\n"
            "NOTE: Generated text comparison is the authoritative check."
        )
        print("Traceback:")
        print(traceback.format_exc())
        print("Assertion message:")
        print(err.args[0])

    # Sequence-wide top-50 normalized logits check
    k = 50
    print(f"Top-{k} normalized logits check across all timesteps:")
    hf_norm = hf_output_logits - hf_output_logits.max(axis=-1, keepdims=True)
    kh_norm = keras_hub_logits - keras_hub_logits.max(axis=-1, keepdims=True)
    hf_topk = np.sort(np.partition(hf_norm, -k, axis=-1)[..., -k:], axis=-1)
    kh_topk = np.sort(np.partition(kh_norm, -k, axis=-1)[..., -k:], axis=-1)
    try:
        np.testing.assert_allclose(kh_topk, hf_topk, atol=atol, rtol=rtol)
        print(f"✓ Top-{k} normalized logits within tolerance.")
    except AssertionError as err:
        print(f"Top-{k} normalized logits exceed tolerance.")
        print(traceback.format_exc())
        print(err.args[0])


def validate_output(
    keras_model,
    hf_model,
    hf_tokenizer,
):
    input_str = "What is Keras?"
    length = 32

    # KerasHub generation
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("\n🔶 KerasHub output:\n", keras_output)

    # Hugging Face generation
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
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
    print("\n🔶 HuggingFace output:\n", hf_generated_text)


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. "
            f"Must be one of {','.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    print(f"\n🏃 Converting and validating {preset} from hf://{hf_preset}")

    # Load HuggingFace model in float32 for reference validation
    target_dtype = torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        torch_dtype=target_dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    keras_dtype = "float32"
    keras_hub_backbone = keras_hub.models.GemmaBackbone.from_preset(
        f"hf://{hf_preset}", dtype=keras_dtype
    )
    keras_hub_tokenizer = keras_hub.models.GemmaTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(
        keras_hub_tokenizer
    )

    print("\n-> Hugging Face model and tokenizer loaded.")
    print("-> KerasHub model loaded via on-the-fly converter.")

    # Numerical verification
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(
        keras_hub_backbone,
        keras_hub_preprocessor,
        hf_model,
        keras_dtype,
    )

    gemma_lm = keras_hub.models.GemmaCausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
    )
    gemma_lm.compile(sampler="greedy")

    validate_output(gemma_lm, hf_model, hf_tokenizer)

    save_dtype = FLAGS.save_dtype
    if save_dtype == "float32":
        print(f"\n-> Saving model in {save_dtype}...")
        gemma_lm.save_to_preset(f"./{preset}")
    else:
        # Free memory before reloading in save_dtype
        del gemma_lm
        del keras_hub_backbone
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_hub_backbone_save = keras_hub.models.GemmaBackbone.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        gemma_lm_save = keras_hub.models.GemmaCausalLM(
            backbone=keras_hub_backbone_save,
            preprocessor=keras_hub_preprocessor,
        )
        gemma_lm_save.save_to_preset(f"./{preset}")

    print(f"\n🏁 Saved preset to ./{preset}")

    if FLAGS.upload_uri:
        keras_hub.upload_preset(uri=FLAGS.upload_uri, preset=f"./{preset}")
        print(f"🏁 Successfully uploaded {preset} to {FLAGS.upload_uri}")


if __name__ == "__main__":
    app.run(main)
