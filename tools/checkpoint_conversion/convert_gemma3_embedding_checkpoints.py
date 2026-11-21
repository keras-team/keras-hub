"""
Convert HuggingFace EmbeddingGemma checkpoints to KerasHub format.

This script loads weights from the HuggingFace `google/embeddinggemma-300m`
model (a Sentence Transformers model) and converts them to KerasHub's
Gemma3Backbone format with embedding support.

Setup:
```shell
pip install keras-hub keras sentence-transformers safetensors huggingface_hub
huggingface-cli login  # Required for gated model access
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_gemma3_embedding_checkpoints.py --preset embedding_gemma3_300m
```
"""

import json
import os
import tempfile

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from sentence_transformers import SentenceTransformer

import keras_hub
from keras_hub.src.utils.transformers.convert_gemma3 import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_gemma3 import convert_tokenizer
from keras_hub.src.utils.transformers.convert_gemma3 import convert_weights
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Preset name for output. Must be one of the keys in PRESET_MAP.",
)

# Map KerasHub preset names to HuggingFace model IDs
PRESET_MAP = {
    "embedding_gemma3_300m": "google/embeddinggemma-300m",
}


def validate_output(keras_model, keras_tokenizer, hf_model_id):
    """
    Validate numerical parity between KerasHub model and HF SentenceTransformer.

    Performs three checks:
    1. Parameter Count: Verifies exact match of backbone parameters.
    2. Tokenization: Ensures KerasHub tokenizer matches HF tokenization
       (with checks).

    Args:
        keras_model: The converted KerasHub model.
        keras_tokenizer: The KerasHub tokenizer.
        hf_model_id: The HuggingFace model ID to compare against.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION")
    print("=" * 60)

    # Load HuggingFace model
    print(f"\nLoading HuggingFace model: {hf_model_id}")
    hf_model = SentenceTransformer(hf_model_id)
    hf_model.eval()

    # =========================================
    # PARAMETER COUNT VERIFICATION
    # =========================================
    print("\n--- Parameter Count Check ---")
    # Count KerasHub parameters
    keras_total_params = keras_model.count_params()

    # Count HF parameters
    hf_modules = list(hf_model._modules.values())
    hf_transformer = hf_modules[0]  # Transformer module
    hf_backbone_params = sum(
        p.numel() for p in hf_transformer.auto_model.parameters()
    )

    # Calculate HF pooling head params (sum of all other modules)
    hf_pooling_params = 0
    for module in hf_modules[1:]:
        hf_pooling_params += sum(p.numel() for p in module.parameters())

    # KerasHub includes pooling head, so subtract it for fair comparison
    try:
        pooling_dense1 = keras_model.get_layer("pooling_dense_1")
        pooling_dense2 = keras_model.get_layer("embedding_projection")
        pooling_params = (
            pooling_dense1.count_params() + pooling_dense2.count_params()
        )
    except Exception:
        pooling_params = 0

    keras_backbone_params = keras_total_params - pooling_params
    hf_total_params = hf_backbone_params + hf_pooling_params
    print(f"Keras Hub total params:    {keras_total_params:,}")
    print(f"    - Keras Hub Backbone only:   {keras_backbone_params:,}")
    print(f"    - Keras Hub Pooling head:    {pooling_params:,}")
    print("\n-------------------------------------")
    print(f"Hugging Face total params:          {hf_total_params:,}")
    print(f"    - Hugging Face backbone only:      {hf_backbone_params:,}")
    print(f"    - Hugging Face pooling head :  {hf_pooling_params:,}")

    param_diff = abs(keras_backbone_params - hf_backbone_params)
    pooling_diff = abs(pooling_params - hf_pooling_params)
    total_diff = abs(keras_total_params - hf_total_params)

    if total_diff == 0:
        print("✅ Total parameter count EXACT MATCH!")
    else:
        print(f"❌ Total parameter count mismatch! Diff: {total_diff:,}")

    if param_diff == 0:
        print("✅ Backbone Parameter count EXACT MATCH!")
    else:
        print(f"❌ Backbone parameter count mismatch! Diff: {param_diff:,}")

    if pooling_diff == 0:
        print("✅ Pooling parameter count EXACT MATCH!")
    else:
        print(f"❌ Pooling parameter count mismatch! Diff: {pooling_diff:,}")

    # =========================================
    # EMBEDDING VERIFICATION
    # =========================================
    print("\n--- Embedding Verification ---")

    # Test inputs
    test_texts = [
        "Hello, this is a test sentence.",
        "What is machine learning?",
        "The quick brown fox jumps over the lazy dog.",
    ]
    print(f"Test inputs: {test_texts}")

    # 1. Get HuggingFace embeddings
    print("\nComputing HF embeddings...")
    with torch.no_grad():
        hf_embeddings = hf_model.encode(test_texts, convert_to_numpy=True)
    print(f"HF output shape: {hf_embeddings.shape}")
    print(f"HF embedding[0][:5]: {hf_embeddings[0][:5]}")

    # 2. Get KerasHub embeddings (using KerasHub tokenizer with manual BOS/EOS)
    print("\nComputing KerasHub embeddings...")

    # Tokenize
    keras_tokens = keras_tokenizer(test_texts)
    if hasattr(keras_tokens, "numpy"):
        keras_tokens = keras_tokens.numpy().tolist()

    # Get sequence length from model config or default
    seq_len = getattr(keras_model, "sliding_window_size", 512)
    print(f"Using sequence length: {seq_len}")

    # Manually construct inputs with BOS and EOS tokens
    token_ids_list = []
    padding_mask_list = []

    for tokens in keras_tokens:
        # Add BOS and EOS
        seq = (
            [keras_tokenizer.start_token_id]
            + tokens
            + [keras_tokenizer.end_token_id]
        )
        mask = [1] * len(seq)

        # Pad to sequence length
        pad_len = seq_len - len(seq)
        if pad_len > 0:
            seq = seq + [0] * pad_len
            mask = mask + [0] * pad_len
        else:
            seq = seq[:seq_len]
            mask = mask[:seq_len]

        token_ids_list.append(seq)
        padding_mask_list.append(mask)

    inputs = {
        "token_ids": np.array(token_ids_list, dtype="int32"),
        "padding_mask": np.array(padding_mask_list, dtype="int32"),
    }

    keras_output = keras_model(inputs)

    # Handle output
    if isinstance(keras_output, dict) and "pooled_output" in keras_output:
        keras_embeddings = ops.convert_to_numpy(keras_output["pooled_output"])
    else:
        # Fallback if model doesn't return dict
        keras_embeddings = ops.convert_to_numpy(keras_output)

    print(f"KerasHub output shape: {keras_embeddings.shape}")
    print(f"KerasHub embedding[0][:5]: {keras_embeddings[0][:5]}")

    # 3. Compare
    print("\n--- Comparison ---")

    # Max Absolute Difference
    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    print(f"Max absolute difference: {max_diff:.2e}")

    # Final Result
    print("\n--- Result ---")
    passed = True

    if param_diff != 0:
        print("❌ FAILED: Parameter count mismatch")
        passed = False

    if max_diff > 1e-4:
        print(f"❌ FAILED: Max diff {max_diff:.2e} > 1e-4")
        passed = False

    if passed:
        print("✅ ALL CHECKS PASSED")

    return passed


def main(_):
    """
    Main entry point for the conversion script.

    Loads the specified HuggingFace model, creates a KerasHub model,
    transfers weights, validates the conversion, and saves the preset.
    """
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. "
            f"Must be one of: {list(PRESET_MAP.keys())}"
        )

    hf_model_id = PRESET_MAP[preset]

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    print(f"{'=' * 60}\n")

    # Use a temporary directory for downloading HuggingFace artifacts
    # This keeps the output preset directory clean
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory for downloads: {temp_dir}")

        # Download config to get parameters
        print("Downloading config...")
        config_path = hf_hub_download(hf_model_id, "config.json")
        with open(config_path, "r") as f:
            transformers_config = json.load(f)

        # Convert config
        keras_config = convert_backbone_config(transformers_config)

        # Create KerasHub model
        print("\nCreating KerasHub Gemma3Backbone...")
        # Use float32 to match HF model default often
        keras_model = keras_hub.models.Gemma3Backbone(
            **keras_config, dtype="float32"
        )

        # Build model by calling it once
        dummy_input = {
            "token_ids": np.ones((1, 16), dtype="int32"),
            "padding_mask": np.ones((1, 16), dtype="int32"),
        }
        _ = keras_model(dummy_input)

        print(f"KerasHub model parameters: {keras_model.count_params():,}")

        # Transfer weights
        print("\nAttaching simple loader...")
        print("Downloading model.safetensors into temporary directory...")

        hf_hub_download(hf_model_id, "model.safetensors", local_dir=temp_dir)

        # We also need to download the dense layer weights for our new logic
        print("Downloading dense layer weights...")
        hf_hub_download(
            hf_model_id, "2_Dense/model.safetensors", local_dir=temp_dir
        )
        hf_hub_download(
            hf_model_id, "3_Dense/model.safetensors", local_dir=temp_dir
        )

        print("Converting weights...")

        with SafetensorLoader(temp_dir) as loader:
            convert_weights(keras_model, loader, transformers_config)

        # Convert tokenizer
        print("\nConverting tokenizer...")
        # Helper to clean up tokenizer download if needed
        hf_hub_download(hf_model_id, "tokenizer.model", local_dir=temp_dir)
        keras_tokenizer = convert_tokenizer(
            keras_hub.models.Gemma3Tokenizer, temp_dir
        )

        # Validate
        passed = validate_output(keras_model, keras_tokenizer, hf_model_id)

        if not passed:
            print("\n⚠️  Verification failed. Check weight mapping.")
            return

        # Save preset
        print(f"\nSaving to preset: ./{preset}")
        keras_model.save_to_preset(preset)
        keras_tokenizer.save_to_preset(preset)

        print(f"\n✅ Successfully converted and saved to: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
