import argparse
import os

import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
import torch
from transformers import Qwen2MoeForCausalLM

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm import (
    Qwen3_5MoeCausalLM,
)
from keras_hub.src.utils.transformers.export.hf_exporter import export_backbone


def main(args):
    print("Loading KerasHub Qwen3_5Moe model...")
    keras_model = Qwen3_5MoeCausalLM.from_preset(args.preset)

    print(f"Exporting to {args.export_dir}...")
    export_backbone(keras_model.backbone, args.export_dir)

    print("Loading HuggingFace Qwen2Moe model from exported weights...")
    hf_model = Qwen2MoeForCausalLM.from_pretrained(args.export_dir)
    hf_model.eval()

    print("Running equivalence checks...")
    # Dummy input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids).logits

    keras_outputs = keras_model.predict(input_ids.numpy(), verbose=0)

    diff = np.max(np.abs(hf_outputs.numpy() - keras_outputs))
    print(f"Maximum logit difference: {diff}")

    if diff < 1e-4:
        print("✅ Export successful: Outputs match!")
    else:
        print("❌ Export failed: Outputs differ significantly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", type=str, required=True, help="KerasHub preset to convert"
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        required=True,
        help="Directory to save HF weights",
    )
    args = parser.parse_args()
    main(args)
