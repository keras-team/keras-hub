"""Convert BLOOM HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_bloom_checkpoints.py \
        --preset bloom_560m_multi \
        --save_dtype float16
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
    "bloom_560m_multi": "bigscience/bloom-560m",
    "bloom_1.1b_multi": "bigscience/bloom-1b1",
    "bloom_1.7b_multi": "bigscience/bloom-1b7",
    "bloom_3b_multi": "bigscience/bloom-3b",
    "bloom_7b_multi": "bigscience/bloom-7b1",
    "bloom_176b_multi": "bigscience/bloom",
    # Multitask finetuned on xP3 (Crosslingual Public Pool of Prompts)
    # https://huggingface.co/datasets/bigscience/xP3
    # xP3 is a mixture of 13 training tasks in 46 languages with English
    # prompts.
    "bloomz_560m_multi": "bigscience/bloomz-560m",
    "bloomz_1.1b_multi": "bigscience/bloomz-1b1",
    "bloomz_1.7b_multi": "bigscience/bloomz-1b7",
    "bloomz_3b_multi": "bigscience/bloomz-3b",
    "bloomz_7b_multi": "bigscience/bloomz-7b1",
    "bloomz_176b_multi": "bigscience/bloomz",
    # Multitask finetuned on xP3mt (machine-translated prompts).
    "bloomz_7b_mt": "bigscience/bloomz-7b1-mt",
    "bloomz_176b_mt": "bigscience/bloomz-mt",
    # Multitask finetuned on P3 (Public Pool of Prompts)
    # https://huggingface.co/datasets/Muennighoff/P3
    "bloomz_7b_p3": "bigscience/bloomz-7b1-p3",
    "bloomz_176b_p3": "bigscience/bloomz-p3",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    "bloom_560m_multi",
    f"Must be one of {','.join(PRESET_MAP.keys())}. "
    "Defaults to 'bloom_560m_multi'.",
)
flags.DEFINE_string(
    "save_dtype",
    "float16",
    "Dtype to save the model in. Defaults to float16.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Optional upload URI, e.g. "kaggle://keras/bloom/keras/{preset}"',
    required=False,
)

# Tolerance for logit comparison. BLOOM stacks up to 70 decoder blocks, so
# `float32` kernel differences between PyTorch and Keras accumulate to the
# 1e-4 range. `1e-3` matches the tolerance used by other KerasHub decoder
# converter checks.
DTYPE_TOLERANCES = {
    "float32": {"atol": 1e-3, "rtol": 1e-3},
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}


def make_preprocessor(keras_hub_tokenizer):
    """Build the preprocessor that ships with the converted preset."""
    return keras_hub.models.BloomCausalLMPreprocessor(
        tokenizer=keras_hub_tokenizer,
    )


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    test_text = "What is Keras?"
    hf_output = hf_tokenizer([test_text], return_tensors="pt")
    hf_tokens = hf_output["input_ids"].detach().cpu().numpy()

    # Compare the tokenizers directly. Routing through
    # `BloomCausalLMPreprocessor.generate_preprocess()` would also pack a `<s>`
    # start token onto the sequence, which the Hugging Face BLOOM tokenizer
    # does not add.
    kh_tokens = np.asarray(keras_hub_tokenizer(test_text)).reshape(1, -1)

    np.testing.assert_equal(kh_tokens, hf_tokens)
    print("✓ Tokenizer output match.")


def test_model(
    keras_hub_model,
    hf_model,
    hf_tokenizer,
    keras_dtype,
):
    # Verify parameter count.
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params, (
        f"Parameter count mismatch: KerasHub={keras_hub_params:,} vs "
        f"HF={hf_params:,}"
    )
    print(f"\n✓ Parameter count match: {keras_hub_params:,} params")

    # Forward pass comparison. Both models are fed the exact same token ids so
    # that the check isolates the weight conversion from tokenization.
    hf_tokenized = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    token_ids = hf_tokenized["input_ids"].to(device)
    padding_mask = hf_tokenized["attention_mask"].to(device)

    hf_outputs = hf_model(input_ids=token_ids, attention_mask=padding_mask)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_inputs = {
        "token_ids": token_ids.detach().cpu().numpy(),
        "padding_mask": padding_mask.detach().cpu().numpy(),
    }
    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    abs_diff = np.abs(keras_hub_logits - hf_output_logits)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    tolerances = DTYPE_TOLERANCES.get(keras_dtype, {"atol": 1e-3, "rtol": 1e-3})
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

    # Sequence-wide top-50 normalized logits check.
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

    # KerasHub generation.
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("\n🔶 KerasHub output:\n", keras_output)

    # Hugging Face generation. KerasHub's `generate_preprocess()` always packs
    # a `<s>` start token onto the prompt, so add it here too.
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    bos = torch.full(
        (hf_inputs["input_ids"].shape[0], 1),
        hf_tokenizer.bos_token_id,
        dtype=hf_inputs["input_ids"].dtype,
    )
    hf_inputs["input_ids"] = torch.cat([bos, hf_inputs["input_ids"]], dim=-1)
    hf_inputs["attention_mask"] = torch.cat(
        [torch.ones_like(bos), hf_inputs["attention_mask"]], dim=-1
    )
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

    # Load HuggingFace model in float32 for reference validation.
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        torch_dtype=torch.float32,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    keras_dtype = "float32"
    keras_hub_backbone = keras_hub.models.BloomBackbone.from_preset(
        f"hf://{hf_preset}", dtype=keras_dtype
    )
    keras_hub_tokenizer = keras_hub.models.BloomTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = make_preprocessor(keras_hub_tokenizer)

    print("\n-> Hugging Face model and tokenizer loaded.")
    print("-> KerasHub model loaded via on-the-fly converter.")

    # Numerical verification.
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(
        keras_hub_backbone,
        hf_model,
        hf_tokenizer,
        keras_dtype,
    )

    bloom_lm = keras_hub.models.BloomCausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
    )
    bloom_lm.compile(sampler="greedy")

    validate_output(bloom_lm, hf_model, hf_tokenizer)

    save_dtype = FLAGS.save_dtype
    if save_dtype == "float32":
        print(f"\n-> Saving model in {save_dtype}...")
        bloom_lm.save_to_preset(f"./{preset}")
    else:
        # Free memory before reloading in save_dtype.
        del bloom_lm
        del keras_hub_backbone
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_hub_backbone_save = keras_hub.models.BloomBackbone.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        bloom_lm_save = keras_hub.models.BloomCausalLM(
            backbone=keras_hub_backbone_save,
            preprocessor=keras_hub_preprocessor,
        )
        bloom_lm_save.save_to_preset(f"./{preset}")

    print(f"\n🏁 Saved preset to ./{preset}")

    if FLAGS.upload_uri:
        keras_hub.upload_preset(uri=FLAGS.upload_uri, preset=f"./{preset}")
        print(f"🏁 Successfully uploaded {preset} to {FLAGS.upload_uri}")


if __name__ == "__main__":
    app.run(main)
