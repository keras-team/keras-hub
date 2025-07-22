"""
Script to convert LayoutLMv3 checkpoints from Hugging Face to Keras format.
"""

import json
import os

import keras
import numpy as np
from transformers import LayoutLMv3Config
from transformers import LayoutLMv3Model as HFLayoutLMv3Model
from transformers import LayoutLMv3Tokenizer as HFLayoutLMv3Tokenizer

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import (
    LayoutLMv3Tokenizer,
)


def convert_checkpoint(
    hf_model_name_or_path,
    output_dir,
    model_size="base",
):
    """Convert a LayoutLMv3 checkpoint from Hugging Face to Keras format."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Hugging Face model: {hf_model_name_or_path}")

    # Load Hugging Face model, config and tokenizer
    hf_model = HFLayoutLMv3Model.from_pretrained(hf_model_name_or_path)
    hf_config = LayoutLMv3Config.from_pretrained(hf_model_name_or_path)
    hf_tokenizer = HFLayoutLMv3Tokenizer.from_pretrained(hf_model_name_or_path)

    # Get spatial embedding dimensions from the model
    hf_weights = hf_model.state_dict()

    # Check if spatial projection weights exist in the model
    spatial_projections = {}
    for coord in ["x", "y", "h", "w"]:
        proj_key = f"embeddings.{coord}_position_proj.weight"
        if proj_key in hf_weights:
            spatial_projections[coord] = hf_weights[proj_key].numpy()
            shape = spatial_projections[coord].shape
            print(f"Found {coord} projection weights: {shape}")
        else:
            print(f"Warning: {proj_key} not found in model weights")

    # Get spatial embedding dimensions
    x_dim = hf_weights["embeddings.x_position_embeddings.weight"].shape[1]
    y_dim = hf_weights["embeddings.y_position_embeddings.weight"].shape[1]
    h_dim = hf_weights["embeddings.h_position_embeddings.weight"].shape[1]
    w_dim = hf_weights["embeddings.w_position_embeddings.weight"].shape[1]

    # Use maximum dimension for all spatial embeddings
    spatial_embedding_dim = max(x_dim, y_dim, h_dim, w_dim)

    print(f"\nModel: {hf_model_name_or_path}")
    print("Spatial embedding dimensions:")
    print(f"x: {x_dim}, y: {y_dim}, h: {h_dim}, w: {w_dim}")
    print(f"Using dimension: {spatial_embedding_dim}")

    # Create Keras model with correct configuration
    keras_model = LayoutLMv3Backbone(
        vocabulary_size=hf_config.vocab_size,
        hidden_dim=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        dropout=hf_config.hidden_dropout_prob,
        max_sequence_length=hf_config.max_position_embeddings,
        type_vocab_size=hf_config.type_vocab_size,
        initializer_range=hf_config.initializer_range,
        layer_norm_epsilon=hf_config.layer_norm_eps,
        spatial_embedding_dim=spatial_embedding_dim,
        dtype="float32",
    )

    # Create dummy inputs to build the model
    batch_size = 2
    seq_len = 512

    dummy_inputs = {
        "token_ids": keras.ops.ones((batch_size, seq_len), dtype="int32"),
        "padding_mask": keras.ops.ones((batch_size, seq_len), dtype="int32"),
        "bbox": keras.ops.ones((batch_size, seq_len, 4), dtype="int32"),
    }

    # Build the model
    print("Building Keras model...")
    _ = keras_model(dummy_inputs)
    print("Model built successfully")

    print("\nTransferring weights...")

    # Word embeddings
    keras_model.token_embedding.embeddings.assign(
        hf_weights["embeddings.word_embeddings.weight"].numpy()
    )
    print("✓ Word embeddings")

    # Position embeddings
    keras_model.position_embedding.embeddings.assign(
        hf_weights["embeddings.position_embeddings.weight"].numpy()
    )
    print("✓ Position embeddings")

    # Spatial embeddings
    x_weights = hf_weights["embeddings.x_position_embeddings.weight"].numpy()
    y_weights = hf_weights["embeddings.y_position_embeddings.weight"].numpy()
    h_weights = hf_weights["embeddings.h_position_embeddings.weight"].numpy()
    w_weights = hf_weights["embeddings.w_position_embeddings.weight"].numpy()

    # Pad smaller embeddings to match the maximum dimension
    if h_dim < spatial_embedding_dim:
        h_weights = np.pad(
            h_weights,
            ((0, 0), (0, spatial_embedding_dim - h_dim)),
            mode="constant",
            constant_values=0,
        )
        print(f"✓ Padded h_weights from {h_dim} to {spatial_embedding_dim}")

    if w_dim < spatial_embedding_dim:
        w_weights = np.pad(
            w_weights,
            ((0, 0), (0, spatial_embedding_dim - w_dim)),
            mode="constant",
            constant_values=0,
        )
        print(f"✓ Padded w_weights from {w_dim} to {spatial_embedding_dim}")

    # Set spatial embedding weights
    keras_model.x_position_embedding.embeddings.assign(x_weights)
    keras_model.y_position_embedding.embeddings.assign(y_weights)
    keras_model.h_position_embedding.embeddings.assign(h_weights)
    keras_model.w_position_embedding.embeddings.assign(w_weights)
    print("✓ Spatial position embeddings")

    # Load spatial projection weights if available, otherwise initialize
    for coord in ["x", "y", "h", "w"]:
        projection_layer = getattr(keras_model, f"{coord}_projection")

        if coord in spatial_projections:
            # Load actual weights from HF model
            weight_matrix = spatial_projections[coord].T  # Transpose for Keras
            bias_vector = np.zeros(hf_config.hidden_size)
            projection_layer.set_weights([weight_matrix, bias_vector])
            print(f"✓ Loaded {coord} projection weights from HF model")
        else:
            # Initialize with proper dimensions if not found in HF model
            weight_matrix = np.random.normal(
                0,
                hf_config.initializer_range,
                (spatial_embedding_dim, hf_config.hidden_size),
            )
            bias_vector = np.zeros(hf_config.hidden_size)
            projection_layer.set_weights([weight_matrix, bias_vector])
            print(f"⚠ Initialized {coord} projection weights randomly")

    # Token type embeddings
    keras_model.token_type_embedding.embeddings.assign(
        hf_weights["embeddings.token_type_embeddings.weight"].numpy()
    )
    print("✓ Token type embeddings")

    # Embeddings layer normalization
    keras_model.embeddings_layer_norm.set_weights(
        [
            hf_weights["embeddings.LayerNorm.weight"].numpy(),
            hf_weights["embeddings.LayerNorm.bias"].numpy(),
        ]
    )
    print("✓ Embeddings layer norm")

    # Transformer layers
    for i in range(hf_config.num_hidden_layers):
        layer = keras_model.transformer_layers[i]

        # Multi-head attention
        # Note: TransformerEncoder uses different weight naming
        # Map HF attention weights to Keras TransformerEncoder weights

        # Query, Key, Value weights (combined in TransformerEncoder)
        q_weight = (
            hf_weights[f"encoder.layer.{i}.attention.self.query.weight"]
            .numpy()
            .T
        )
        q_bias = hf_weights[
            f"encoder.layer.{i}.attention.self.query.bias"
        ].numpy()
        k_weight = (
            hf_weights[f"encoder.layer.{i}.attention.self.key.weight"].numpy().T
        )
        k_bias = hf_weights[
            f"encoder.layer.{i}.attention.self.key.bias"
        ].numpy()
        v_weight = (
            hf_weights[f"encoder.layer.{i}.attention.self.value.weight"]
            .numpy()
            .T
        )
        v_bias = hf_weights[
            f"encoder.layer.{i}.attention.self.value.bias"
        ].numpy()

        # Note: Individual weights are used separately for TransformerEncoder

        layer._self_attention_layer._query_dense.set_weights([q_weight, q_bias])
        layer._self_attention_layer._key_dense.set_weights([k_weight, k_bias])
        layer._self_attention_layer._value_dense.set_weights([v_weight, v_bias])

        # Output projection
        out_weight = (
            hf_weights[f"encoder.layer.{i}.attention.output.dense.weight"]
            .numpy()
            .T
        )
        out_bias = hf_weights[
            f"encoder.layer.{i}.attention.output.dense.bias"
        ].numpy()
        layer._self_attention_layer._output_dense.set_weights(
            [out_weight, out_bias]
        )

        # Attention layer norm
        attn_norm_weight = hf_weights[
            f"encoder.layer.{i}.attention.output.LayerNorm.weight"
        ].numpy()
        attn_norm_bias = hf_weights[
            f"encoder.layer.{i}.attention.output.LayerNorm.bias"
        ].numpy()
        layer._self_attention_layernorm.set_weights(
            [attn_norm_weight, attn_norm_bias]
        )

        # Feed forward network
        ff1_weight = (
            hf_weights[f"encoder.layer.{i}.intermediate.dense.weight"].numpy().T
        )
        ff1_bias = hf_weights[
            f"encoder.layer.{i}.intermediate.dense.bias"
        ].numpy()
        layer._feedforward_intermediate_dense.set_weights(
            [ff1_weight, ff1_bias]
        )

        ff2_weight = (
            hf_weights[f"encoder.layer.{i}.output.dense.weight"].numpy().T
        )
        ff2_bias = hf_weights[f"encoder.layer.{i}.output.dense.bias"].numpy()
        layer._feedforward_output_dense.set_weights([ff2_weight, ff2_bias])

        # Feed forward layer norm
        ff_norm_weight = hf_weights[
            f"encoder.layer.{i}.output.LayerNorm.weight"
        ].numpy()
        ff_norm_bias = hf_weights[
            f"encoder.layer.{i}.output.LayerNorm.bias"
        ].numpy()
        layer._feedforward_layernorm.set_weights([ff_norm_weight, ff_norm_bias])

        print(f"✓ Transformer layer {i}")

    print("\nWeight transfer completed successfully!")

    # Save the model
    model_path = os.path.join(output_dir, f"layoutlmv3_{model_size}.keras")
    keras_model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    # Create and save tokenizer
    vocab = dict(hf_tokenizer.get_vocab())
    keras_tokenizer = LayoutLMv3Tokenizer(vocabulary=vocab)

    # Save tokenizer
    tokenizer_config = keras_tokenizer.get_config()
    tokenizer_path = os.path.join(
        output_dir, f"layoutlmv3_{model_size}_tokenizer.json"
    )
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Tokenizer config saved to {tokenizer_path}")

    # Save model configuration
    model_config = keras_model.get_config()
    config_path = os.path.join(
        output_dir, f"layoutlmv3_{model_size}_config.json"
    )
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Model config saved to {config_path}")

    print(
        f"\n✅ Successfully converted {hf_model_name_or_path} to Keras format"
    )
    print(f"📁 All files saved to {output_dir}")


def main():
    """Convert LayoutLMv3 checkpoints."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LayoutLMv3 checkpoints"
    )
    parser.add_argument(
        "--model",
        default="microsoft/layoutlmv3-base",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/layoutlmv3",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["base", "large"],
        help="Model size identifier",
    )

    args = parser.parse_args()

    try:
        convert_checkpoint(
            args.model,
            args.output_dir,
            args.model_size,
        )
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
