"""Qwen3.5-MoE KerasHub → HuggingFace Export Verification Script.

Performs a real-weights round-trip check:

  1. Load a KerasHub Qwen3.5-MoE preset and export it to HF format.
  2. Reload the exported checkpoint with
     `Qwen3_5MoeForConditionalGeneration` and report any missing or
     unexpected weight keys.
  3. Compare KerasHub and exported-HF logits on a text prompt.
  4. Optionally compare greedy generations.

Usage:
    KERAS_BACKEND=torch python3 \\
        tools/checkpoint_export/verify_qwen3_5_moe_export.py \\
        --preset qwen3_5_moe_35b_a3b_base \\
        --export_dir /tmp/qwen3_5_moe_hf

Requirements:
    pip install keras-hub transformers torch safetensors
"""

import argparse
import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm import (  # noqa: E402
    Qwen3_5MoeCausalLM,
)

TEXT_PROMPT = "The capital of France is"


def export_keras_model(preset, export_dir):
    """Load a KerasHub preset and export it to HuggingFace format."""
    print(f"\n[1/4] Loading KerasHub model from preset '{preset}'...")
    keras_model = Qwen3_5MoeCausalLM.from_preset(preset)
    backbone = keras_model.backbone
    print(
        f"  ✓ Loaded: {backbone.num_layers} layers, {backbone.hidden_dim}d, "
        f"{backbone.num_experts} experts (top-{backbone.top_k}), "
        f"{backbone.vocabulary_size} vocab"
    )
    print(f"  ✓ Layer types: {backbone.layer_types}")

    print(f"\n[2/4] Exporting to HF format → {export_dir}...")
    keras_model.export_to_transformers(export_dir)
    for fname in ["config.json", "tokenizer_config.json"]:
        fpath = os.path.join(export_dir, fname)
        print(
            f"  {'✓' if os.path.exists(fpath) else '✗'} {fname} "
            f"({os.path.getsize(fpath) if os.path.exists(fpath) else 0:,} B)"
        )
    return keras_model


def load_exported_model(export_dir):
    """Reload the exported checkpoint and report weight-key mismatches."""
    print("\n[3/4] Loading exported checkpoint with Transformers...")
    hf_model, loading_info = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        export_dir,
        dtype=torch.float32,
        output_loading_info=True,
    )
    hf_model.eval()

    missing = [
        k
        for k in loading_info.get("missing_keys", [])
        # `visual` is absent for text-only exports.
        if not k.startswith("model.visual")
    ]
    unexpected = loading_info.get("unexpected_keys", [])
    print(f"  Missing keys:    {len(missing)}")
    for key in missing[:10]:
        print(f"    - {key}")
    print(f"  Unexpected keys: {len(unexpected)}")
    for key in unexpected[:10]:
        print(f"    - {key}")
    if missing or unexpected:
        print("  ✗ Weight map is incomplete or mis-named.")
    else:
        print("  ✓ All weights mapped.")
    return hf_model, not (missing or unexpected)


def compare_logits(keras_model, hf_model, export_dir, tolerance):
    """Compare KerasHub and exported-HF logits on a text prompt."""
    print("\n[4/4] Comparing logits...")
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    input_ids = np.array([tokenizer(TEXT_PROMPT)["input_ids"]], dtype="int32")

    keras_logits = keras.ops.convert_to_numpy(
        keras_model(
            {
                "token_ids": input_ids,
                "padding_mask": np.ones_like(input_ids),
            }
        )
    )
    with torch.no_grad():
        hf_logits = (
            hf_model(input_ids=torch.tensor(input_ids, dtype=torch.long))
            .logits.float()
            .cpu()
            .numpy()
        )

    diff = float(np.max(np.abs(keras_logits - hf_logits)))
    print(f"  KerasHub logits: {keras_logits.shape}")
    print(f"  HF logits:       {hf_logits.shape}")
    print(f"  Max abs diff:    {diff:.3e} (tolerance {tolerance:.1e})")
    return diff < tolerance


def main(args):
    keras_model = export_keras_model(args.preset, args.export_dir)
    hf_model, weights_ok = load_exported_model(args.export_dir)
    logits_ok = compare_logits(
        keras_model, hf_model, args.export_dir, args.tolerance
    )

    del keras_model, hf_model
    gc.collect()

    print("\n" + "=" * 60)
    if weights_ok and logits_ok:
        print("✅ Export successful: weights mapped and outputs match.")
        return 0
    print("❌ Export verification failed.")
    print(f"   Weight keys: {'ok' if weights_ok else 'MISMATCH'}")
    print(f"   Logits:      {'ok' if logits_ok else 'MISMATCH'}")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="KerasHub Qwen3.5-MoE preset to export.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        required=True,
        help="Directory to write the HuggingFace checkpoint to.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed absolute logit difference.",
    )
    raise SystemExit(main(parser.parse_args()))
