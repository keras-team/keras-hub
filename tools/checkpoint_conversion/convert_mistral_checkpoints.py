"""Convert Mistral HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_mistral_checkpoints.py \
        --preset mistral_7b_en
"""

import gc

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoTokenizer
from transformers import MistralForCausalLM

import keras_hub

PRESET_MAP = {
    "mistral_7b_en": "mistralai/Mistral-7B-v0.1",
    "mistral_0.3_7b_en": "mistralai/Mistral-7B-v0.3",
    "mistral_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral_0.2_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_0.3_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.3",
    "magistral_small_2506_en": "mistralai/Magistral-Small-2506",
    "magistral_small_2507_en": "mistralai/Magistral-Small-2507",
}

TEXT_PROMPT = "What is Keras?"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def precompute_hf_outputs(hf_model, hf_tokenizer):
    """Precompute the HF reference outputs needed for validation.

    Runs the HF forward pass and returns the results as numpy arrays so
    the HF model can be deleted afterwards to free memory.
    """
    hf_inputs = hf_tokenizer([TEXT_PROMPT], return_tensors="pt")
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    return {
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().numpy(),
        "num_parameters": hf_model.num_parameters(),
    }


def validate_output(keras_model, hf_results):
    """Validate parameter count, tokenization and logits against HF."""
    # Parameter count comparison.
    keras_params = keras_model.backbone.count_params()
    hf_params = hf_results["num_parameters"]
    print(f"\nKerasHub params: {keras_params:,}")
    print(f"HF params:       {hf_params:,}")
    assert keras_params == hf_params

    # Token ID parity.
    hf_token_ids = hf_results["token_ids"]
    sequence_length = hf_token_ids.shape[1]
    keras_inputs = keras_model.preprocessor(
        [TEXT_PROMPT], sequence_length=sequence_length
    )[0]
    keras_token_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    np.testing.assert_equal(keras_token_ids, hf_token_ids)
    print("-> Token IDs match.")

    # Logit comparison on the exact HF token ids to verify the weights
    # ported correctly. Both models run in float32, so they should agree
    # closely; porting mistakes produce diffs orders of magnitude larger.
    token_ids = ops.convert_to_tensor(hf_token_ids.astype("int32"))
    padding_mask = ops.ones_like(token_ids)
    keras_hidden = keras_model.backbone(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    keras_logits = keras_model.backbone.token_embedding(
        keras_hidden, reverse=True
    )
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")

    hf_logits = hf_results["logits"]
    abs_diff = np.abs(keras_logits - hf_logits)
    print("KerasHub logits:", keras_logits[0, 0, :5])
    print("HF logits:      ", hf_logits[0, 0, :5])
    print(f"Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"Logit max absolute diff:  {abs_diff.max():.6f}")
    np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-3)
    print("-> Logits match! (atol=1e-3)")


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load the HF model in float32 and precompute reference outputs ===
    # float32 avoids bfloat16 tensors that NumPy cannot handle and gives
    # a precise reference for the numerics check.
    hf_model = MistralForCausalLM.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer)
    print("\n-> Huggingface model loaded and reference outputs computed")

    # === Free the HF model so only one model is in memory at a time ===
    del hf_model, hf_tokenizer
    gc.collect()

    # === Load the KerasHub model in float32 via the built-in converter ===
    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("\n-> KerasHub model loaded")

    # === Check that the model and tokenizer outputs match ===
    validate_output(keras_model, hf_results)
    print("\n-> Tests passed!")

    # === Reload in bfloat16 and save the preset ===
    del keras_model
    gc.collect()
    keras_model = keras_hub.models.MistralCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print("\n-> Saved the model preset in bfloat16")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
