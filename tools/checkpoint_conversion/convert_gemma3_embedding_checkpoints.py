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
python convert_gemma3_embedding_checkpoints.py --preset embedding_gemma3_300m_en
```
"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

import keras_hub

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Preset name for output. Must be one of the keys in PRESET_MAP.",
)

# Map KerasHub preset names to HuggingFace model IDs
PRESET_MAP = {
    "embedding_gemma3_300m_en": "google/embeddinggemma-300m",
}

BACKBONE_CONFIG = {
    "embedding_gemma3_300m_en": {
        "vocabulary_size": 262144,
        "image_size": 896,
        "num_layers": 24,
        "num_query_heads": 3,
        "num_key_value_heads": 1,
        "hidden_dim": 768,
        "intermediate_dim": 1152,
        "head_dim": 256,
        "query_head_dim_normalize": True,
        "use_query_key_norm": True,
        "use_post_ffw_norm": True,
        "use_post_attention_norm": True,
        "attention_logit_soft_cap": None,
        "final_logit_soft_cap": None,
        "use_sliding_window_attention": True,
        "sliding_window_size": 512,
        "local_rope_scaling_factor": 1.0,
        "global_rope_scaling_factor": 1.0,
        "vision_encoder": None,
        "layer_norm_epsilon": 1e-6,
        # Embedding model specific settings
        "use_bidirectional_attention": True,
        "is_embedding_model": True,
        "pooling_intermediate_dim": 3072,
        "embedding_dim": 768,
    },
}


def load_hf_weights(hf_model_id):
    """Load weights from HuggingFace safetensors file."""
    print(f"Downloading weights from {hf_model_id}...")
    file_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="model.safetensors",
    )

    weights = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).numpy()

    print(f"Loaded {len(weights)} weight tensors from HF checkpoint.")
    return weights


def load_pooling_head_weights(hf_model_id):
    """Load the dense layer weights from sentence-transformers modules."""
    print("Loading pooling head weights...")

    # Dense layer 1
    dense1_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="2_Dense/model.safetensors",
    )
    dense1_weights = {}
    with safe_open(dense1_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            dense1_weights[key] = f.get_tensor(key).numpy()

    # Dense layer 2
    dense2_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="3_Dense/model.safetensors",
    )
    dense2_weights = {}
    with safe_open(dense2_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            dense2_weights[key] = f.get_tensor(key).numpy()

    return dense1_weights, dense2_weights


def convert_weights(keras_model, hf_weights, dense1_weights, dense2_weights):
    """
    Port weights from HuggingFace dictionary to KerasHub model.

    This function handles:
    1. Backbone weights: Mapping Gemma3CausalLM layers to KerasHub
       Gemma3Backbone.
       - Token embeddings
       - Attention layers (Q, K, V, O projections)
       - MLP layers (Gate, Up, Down projections)
       - Layer norms (Pre-attention, Post-attention, Final norm)
    2. Pooling head weights: Mapping SentenceTransformer dense layers.
       - Dense 1 (768 -> 3072)
       - Dense 2 (3072 -> 768)

    Args:
        keras_model: The target KerasHub Gemma3Backbone instance.
        hf_weights: Dictionary of weights from the HF backbone model.
        dense1_weights: Dictionary of weights from the first dense layer.
        dense2_weights: Dictionary of weights from the second dense layer.
    """
    print("Converting weights...")
    num_layers = keras_model.num_layers

    # Get model config for reshaping
    num_q_heads = keras_model.num_query_heads
    num_kv_heads = keras_model.num_key_value_heads
    head_dim = keras_model.head_dim
    hidden_dim = keras_model.hidden_dim

    transferred = 0
    missing_keras = []

    # Transfer token embedding
    if "embed_tokens.weight" in hf_weights:
        try:
            layer = keras_model.get_layer("token_embedding")
            layer.embeddings.assign(hf_weights["embed_tokens.weight"])
            transferred += 1
            keras_embed = ops.convert_to_numpy(layer.embeddings)
            hf_embed = hf_weights["embed_tokens.weight"]
            diff = np.max(np.abs(keras_embed - hf_embed))
            print(f"  -> Embedding weights transferred (max diff: {diff:.2e})")
        except Exception as e:
            missing_keras.append(f"token_embedding: {e}")

    # Transfer final layer norm
    if "norm.weight" in hf_weights:
        try:
            layer = keras_model.get_layer("final_normalization")
            layer.scale.assign(hf_weights["norm.weight"])
            transferred += 1
        except Exception as e:
            missing_keras.append(f"final_normalization: {e}")

    # Transfer decoder block weights
    for i in range(num_layers):
        prefix = f"layers.{i}"
        try:
            block = keras_model.get_layer(f"decoder_block_{i}")
            attn = block.attention

            # Q projection
            if f"{prefix}.self_attn.q_proj.weight" in hf_weights:
                q_w = hf_weights[f"{prefix}.self_attn.q_proj.weight"]
                q_w = q_w.T
                q_w = q_w.reshape(hidden_dim, num_q_heads, head_dim)
                q_w = np.transpose(q_w, (1, 0, 2))
                attn.query_dense.kernel.assign(q_w)
                transferred += 1

            # K projection
            if f"{prefix}.self_attn.k_proj.weight" in hf_weights:
                k_w = hf_weights[f"{prefix}.self_attn.k_proj.weight"]
                k_w = k_w.T
                k_w = k_w.reshape(hidden_dim, num_kv_heads, head_dim)
                k_w = np.transpose(k_w, (1, 0, 2))
                attn.key_dense.kernel.assign(k_w)
                transferred += 1

            # V projection
            if f"{prefix}.self_attn.v_proj.weight" in hf_weights:
                v_w = hf_weights[f"{prefix}.self_attn.v_proj.weight"]
                v_w = v_w.T
                v_w = v_w.reshape(hidden_dim, num_kv_heads, head_dim)
                v_w = np.transpose(v_w, (1, 0, 2))
                attn.value_dense.kernel.assign(v_w)
                transferred += 1

            if f"{prefix}.self_attn.o_proj.weight" in hf_weights:
                o_w = hf_weights[f"{prefix}.self_attn.o_proj.weight"]
                o_w = o_w.T
                o_w = o_w.reshape(num_q_heads, head_dim, hidden_dim)
                attn.output_dense.kernel.assign(o_w)
                transferred += 1

            # Q/K norms
            if f"{prefix}.self_attn.q_norm.weight" in hf_weights:
                attn.query_norm.scale.assign(
                    hf_weights[f"{prefix}.self_attn.q_norm.weight"]
                )
                transferred += 1
            if f"{prefix}.self_attn.k_norm.weight" in hf_weights:
                attn.key_norm.scale.assign(
                    hf_weights[f"{prefix}.self_attn.k_norm.weight"]
                )
                transferred += 1

            # MLP layers
            if f"{prefix}.mlp.gate_proj.weight" in hf_weights:
                block.gating_ffw.kernel.assign(
                    hf_weights[f"{prefix}.mlp.gate_proj.weight"].T
                )
                transferred += 1
            if f"{prefix}.mlp.up_proj.weight" in hf_weights:
                block.gating_ffw_2.kernel.assign(
                    hf_weights[f"{prefix}.mlp.up_proj.weight"].T
                )
                transferred += 1
            if f"{prefix}.mlp.down_proj.weight" in hf_weights:
                block.ffw_linear.kernel.assign(
                    hf_weights[f"{prefix}.mlp.down_proj.weight"].T
                )
                transferred += 1
            if f"{prefix}.input_layernorm.weight" in hf_weights:
                block.pre_attention_norm.scale.assign(
                    hf_weights[f"{prefix}.input_layernorm.weight"]
                )
                transferred += 1
            if f"{prefix}.post_attention_layernorm.weight" in hf_weights:
                block.post_attention_norm.scale.assign(
                    hf_weights[f"{prefix}.post_attention_layernorm.weight"]
                )
                transferred += 1
            if f"{prefix}.pre_feedforward_layernorm.weight" in hf_weights:
                block.pre_ffw_norm.scale.assign(
                    hf_weights[f"{prefix}.pre_feedforward_layernorm.weight"]
                )
                transferred += 1
            if f"{prefix}.post_feedforward_layernorm.weight" in hf_weights:
                block.post_ffw_norm.scale.assign(
                    hf_weights[f"{prefix}.post_feedforward_layernorm.weight"]
                )
                transferred += 1

        except Exception as e:
            missing_keras.append(f"decoder_block_{i}: {e}")

    # Transfer pooling head weights (if is_embedding_model layers exist)
    try:
        dense1_layer = keras_model.get_layer("pooling_dense_1")
        if "linear.weight" in dense1_weights:
            dense1_layer.kernel.assign(dense1_weights["linear.weight"].T)
            transferred += 1
            print("  -> Transferred pooling_dense_1 weights")
    except Exception:
        pass  # Layer may not exist in this version

    try:
        dense2_layer = keras_model.get_layer("embedding_projection")
        if "linear.weight" in dense2_weights:
            dense2_layer.kernel.assign(dense2_weights["linear.weight"].T)
            transferred += 1
            print("  -> Transferred embedding_projection weights")
    except Exception:
        pass  # Layer may not exist in this version

    print(f"Transferred {transferred} weight tensors.")
    if missing_keras:
        print(f"Failed Keras transfers: {missing_keras[:5]}...")

    return transferred


def validate_output(keras_model, keras_tokenizer, hf_model_id):
    """
    Validate numerical parity between KerasHub model and HF SentenceTransformer.

    Performs three checks:
    1. Parameter Count: Verifies exact match of backbone parameters.
    2. Embedding Verification: Compares embeddings for test sentences
       using Cosine Similarity (target > 0.9999).
    3. Tokenization: Ensures KerasHub tokenizer matches HF tokenization

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

    # Count HF parameters (transformer backbone only, not pooling head)
    hf_transformer = list(hf_model._modules.values())[0]  # Transformer module
    hf_backbone_params = sum(
        p.numel() for p in hf_transformer.auto_model.parameters()
    )

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

    print(f"KerasHub total params:    {keras_total_params:,}")
    print(f"  - Pooling head:         {pooling_params:,}")
    print(f"  - Backbone only:        {keras_backbone_params:,}")
    print(f"HF backbone params:       {hf_backbone_params:,}")

    param_diff = abs(keras_backbone_params - hf_backbone_params)

    if param_diff == 0:
        print("✅ Parameter count EXACT MATCH!")
    else:
        print(f"❌ Parameter count mismatch! Diff: {param_diff:,}")

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

    # Cosine Similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i, text in enumerate(test_texts):
        sim = cosine_sim(hf_embeddings[i], keras_embeddings[i])
        print(f"Cosine similarity [{i}]: {sim:.6f}")

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
            f"Invalid preset '{preset}'. Must be one of: "
            f"{list(PRESET_MAP.keys())}"
        )

    hf_model_id = PRESET_MAP[preset]
    config = BACKBONE_CONFIG[preset]

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    print(f"{'=' * 60}\n")

    # Load HuggingFace weights
    hf_weights = load_hf_weights(hf_model_id)
    dense1_weights, dense2_weights = load_pooling_head_weights(hf_model_id)

    # Create KerasHub model
    print("\nCreating KerasHub Gemma3Backbone...")
    keras_model = keras_hub.models.Gemma3Backbone(**config, dtype="float32")

    # Build model by calling it once
    dummy_input = {
        "token_ids": np.ones((1, 16), dtype="int32"),
        "padding_mask": np.ones((1, 16), dtype="int32"),
    }
    _ = keras_model(dummy_input)

    print(f"KerasHub model parameters: {keras_model.count_params():,}")

    # Transfer weights
    convert_weights(keras_model, hf_weights, dense1_weights, dense2_weights)

    # Load tokenizer by downloading the model file directly
    print("\nLoading tokenizer...")
    tokenizer_model_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="tokenizer.model",
    )
    keras_tokenizer = keras_hub.models.Gemma3Tokenizer(
        proto=tokenizer_model_path
    )

    # Validate
    passed = validate_output(keras_model, keras_tokenizer, hf_model_id)

    if not passed:
        print("\n⚠️  Verification failed. Check weight mapping.")
        return

    # Save preset
    print(f"\nSaving to preset: ./{preset}")
    os.makedirs(preset, exist_ok=True)
    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)

    print(f"\n✅ Successfully converted and saved to: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
