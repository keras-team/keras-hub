"""Convert GPT-NeoX and Pythia HuggingFace checkpoints to KerasHub presets.

Usage:
    python tools/checkpoint_conversion/convert_gpt_neo_x_checkpoints.py \
        --preset pythia_70m_en
"""

import gc
import os
import random

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "pythia_14m_en": "EleutherAI/pythia-14m",
    "pythia_70m_en": "EleutherAI/pythia-70m",
    "pythia_160m_en": "EleutherAI/pythia-160m",
    "pythia_410m_en": "EleutherAI/pythia-410m",
    "pythia_1.4b_en": "EleutherAI/pythia-1.4b",
    "pythia_2.8b_en": "EleutherAI/pythia-2.8b",
    "pythia_6.9b_en": "EleutherAI/pythia-6.9b",
    "pythia_12b_en": "EleutherAI/pythia-12b",
    "gpt_neox_20b_en": "EleutherAI/gpt-neox-20b",
}

ATOL_MAP = {
    "pythia_14m_en": 0.02,
    "pythia_70m_en": 0.03,
    "pythia_160m_en": 0.01,
    "pythia_410m_en": 0.005,
    "pythia_1.4b_en": 0.003,
    "pythia_2.8b_en": 0.001,
    "pythia_6.9b_en": 0.002,
    "pythia_12b_en": 0.004,
    "gpt_neox_20b_en": 0.002,
}

PROMPTS = [
    "What is Keras?",
    "The quick brown fox jumps over the lazy dog. In 1997, Deep Blue beat "
    "Garry Kasparov, and the history of machine intelligence changed. Since "
    "then, models have grown from millions to hundreds of billions of "
    "parameters, trained on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n\n    return "
    "fibonacci(n - 1) + fibonacci(n - 2)\n\n\nclass Solver:\n    def "
    "__init__(self):\n        self.cache = {}\n",
]
# New tokens generated in the generation smoke test, on both sides.
GENERATE_NEW_TOKENS = 8

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_float(
    "atol",
    None,
    "Max absolute logit difference to accept. Overrides ATOL_MAP.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Optional URI to upload the preset to, e.g. "kaggle://keras/gpt_neox/'
    'keras/pythia_70m_en"',
)


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, hf_tokenizer):
    """Precompute all HF outputs needed for validation."""
    results = []
    for prompt in PROMPTS:
        hf_inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            hf_out = hf_model(**hf_inputs)
            hf_gen = hf_model.generate(
                **hf_inputs,
                max_new_tokens=GENERATE_NEW_TOKENS,
                do_sample=False,
                pad_token_id=hf_tokenizer.eos_token_id,
            )
        results.append(
            {
                "input_ids": hf_inputs["input_ids"]
                .cpu()
                .numpy()
                .astype(np.int32),
                "logits": hf_out.logits.detach().cpu().float().numpy(),
                "generated": hf_tokenizer.decode(
                    hf_gen[0], skip_special_tokens=True
                ),
            }
        )
    return results


# ---------------------------------------------------------------
# 2. Parameter count comparison
# ---------------------------------------------------------------
def test_parameter_count(keras_backbone, hf_param_count):
    """Compare parameter counts between KerasHub and HF models."""
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)

    keras_params = keras_backbone.count_params()
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")

    if keras_params == hf_param_count:
        print("  ✓ Parameter counts match!")
    else:
        diff = hf_param_count - keras_params
        raise AssertionError(f"Parameter count difference: {diff:,}")


# ---------------------------------------------------------------
# 3. Validate tokenization and logits
# ---------------------------------------------------------------
def validate_output(keras_model, hf_results_list, preset):
    """Validate token ids and logits, end to end through KerasHub."""
    print("\n" + "=" * 50)
    print("TOKEN ID AND LOGIT VALIDATION")
    print("=" * 50)

    valid_lens = []
    logit_pairs = []
    max_diffs = []
    for prompt, hf_results in zip(PROMPTS, hf_results_list):
        hf_ids = hf_results["input_ids"]

        # --- Token ID parity ---
        keras_preprocessed = keras_model.preprocessor.generate_preprocess(
            [prompt]
        )
        keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
        keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])
        valid_len = int(np.sum(keras_mask[0]))
        valid_lens.append(valid_len)

        print(f"\n  Prompt: {prompt[:30]}...")
        print(f"  HF token ids:       {hf_ids[0].tolist()}")
        print(f"  KerasHub token ids: {keras_ids[0, :valid_len].tolist()}")
        np.testing.assert_array_equal(keras_ids[0, :valid_len], hf_ids[0])
        print("  ✓ Token IDs match.")

        # --- Logit comparison (task call, KerasHub's own ids) ---
        keras_logits = ops.convert_to_numpy(
            keras_model(keras_preprocessed)
        ).astype(np.float32)[:, :valid_len]
        hf_logits = hf_results["logits"]
        abs_diff = np.abs(keras_logits - hf_logits)
        print(f"  Logit mean absolute diff: {abs_diff.mean():.6f}")
        print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
        logit_pairs.append((keras_logits, hf_logits))
        max_diffs.append(abs_diff.max())

    max_diff = max(max_diffs)
    print(f"\n  Max absolute logit diff over all prompts: {max_diff:.6f}")
    atol = FLAGS.atol if FLAGS.atol is not None else ATOL_MAP.get(preset)
    if atol is None:
        raise ValueError(
            f"ATOL_MAP has no measured tolerance for {preset}. The max "
            f"absolute logit diff over all prompts is {max_diff:.6f}; set "
            "ATOL_MAP just above it, or pass --atol."
        )
    for keras_logits, hf_logits in logit_pairs:
        np.testing.assert_allclose(keras_logits, hf_logits, atol=atol)
    print(f"  ✓ Logits match within atol={atol} on every prompt.")
    return valid_lens


# ---------------------------------------------------------------
# 4. Generation Smoke Test
# ---------------------------------------------------------------
def generation_smoke(keras_model, hf_results_list, prompt_lens):
    print("\n" + "=" * 50)
    print("GENERATION SMOKE TEST")
    print("=" * 50)

    keras_model.compile(sampler="greedy")
    for prompt, hf_results, prompt_len in zip(
        PROMPTS, hf_results_list, prompt_lens
    ):
        keras_output = keras_model.generate(
            [prompt], max_length=prompt_len + GENERATE_NEW_TOKENS
        )[0]
        print(f"\nPrompt: {prompt[:30]}...")
        print(f"HF:       {hf_results['generated']}")
        print(f"KerasHub: {keras_output}")
        assert keras_output == hf_results["generated"], (
            "Generated text does not match HF"
        )
        print("  ✓ Generation matches.")


# ---------------------------------------------------------------
# 5. Save preset
# ---------------------------------------------------------------
def save_preset(hf_preset, preset_name):
    """Save the converted model as a bfloat16 KerasHub preset."""
    print(f"\n-> Saving KerasHub preset to ./{preset_name} in bfloat16...")
    keras_model = keras_hub.models.GPTNeoXCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]

    # --- Phase 1: Load HF model and precompute all outputs ---
    print("-> Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer)
    print("   HF outputs precomputed!")

    # --- Phase 2: Free HF model to reclaim memory ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    print("   HF model released.")

    # --- Phase 3: Load KerasHub model ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.GPTNeoXCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate against precomputed HF outputs ---
    test_parameter_count(keras_model.backbone, hf_params)
    prompt_lens = validate_output(keras_model, hf_results, preset)
    generation_smoke(keras_model, hf_results, prompt_lens)

    # --- Phase 5: Save preset ---
    del keras_model
    gc.collect()
    save_preset(hf_preset, preset)
    if FLAGS.upload_uri:
        keras_hub.upload_preset(FLAGS.upload_uri, f"./{preset}")
        print(f"  ✓ Preset uploaded to {FLAGS.upload_uri}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
