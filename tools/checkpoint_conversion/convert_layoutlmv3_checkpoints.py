"""
Script to convert LayoutLMv3 checkpoints from Hugging Face to Keras format.
"""

import json
import os

import numpy as np
import tensorflow as tf
from transformers import LayoutLMv3Config
from transformers import LayoutLMv3Model as HFLayoutLMv3Model
from transformers import LayoutLMv3Tokenizer as HFLayoutLMv3Tokenizer

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)


def convert_checkpoint(
    hf_model_name_or_path,
    output_dir,
    model_size="base",
):
    """Convert a LayoutLMv3 checkpoint from Hugging Face to Keras format."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Hugging Face model, config and tokenizer
    hf_model = HFLayoutLMv3Model.from_pretrained(hf_model_name_or_path)
    hf_config = LayoutLMv3Config.from_pretrained(hf_model_name_or_path)
    hf_tokenizer = HFLayoutLMv3Tokenizer.from_pretrained(hf_model_name_or_path)

    # Get spatial embedding dimensions from the model
    hf_weights = hf_model.state_dict()
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

    # Create dummy inputs
    batch_size = 2
    seq_len = 512
    input_ids = tf.random.uniform(
        (batch_size, seq_len),
        minval=0,
        maxval=hf_config.vocab_size,
        dtype=tf.int32,
    )
    bbox = tf.random.uniform(
        (batch_size, seq_len, 4), minval=0, maxval=1000, dtype=tf.int32
    )
    attention_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)
    image = tf.random.uniform(
        (batch_size, 112, 112, 3), minval=0, maxval=1, dtype=tf.float32
    )

    # Build the model with dummy inputs
    keras_model = LayoutLMv3Backbone.from_preset(
        f"layoutlmv3_{model_size}",
        input_shape={
            "input_ids": (batch_size, seq_len),
            "bbox": (batch_size, seq_len, 4),
            "attention_mask": (batch_size, seq_len),
            "image": (batch_size, 112, 112, 3),
        },
    )

    # Build model with dummy inputs
    _ = keras_model(
        {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "image": image,
        }
    )

    # Print shapes of spatial embedding weights
    print("\nSpatial embedding shapes:")
    print(
        f"x_position_embeddings: "
        f"{hf_weights['embeddings.x_position_embeddings.weight'].shape}"
    )
    print(
        f"y_position_embeddings: "
        f"{hf_weights['embeddings.y_position_embeddings.weight'].shape}"
    )
    print(
        f"h_position_embeddings: "
        f"{hf_weights['embeddings.h_position_embeddings.weight'].shape}"
    )
    print(
        f"w_position_embeddings: "
        f"{hf_weights['embeddings.w_position_embeddings.weight'].shape}"
    )

    # Word embeddings
    keras_model.word_embeddings.set_weights(
        [hf_weights["embeddings.word_embeddings.weight"].numpy()]
    )

    # Position embeddings
    keras_model.position_embeddings.set_weights(
        [hf_weights["embeddings.position_embeddings.weight"].numpy()]
    )

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
        )
    if w_dim < spatial_embedding_dim:
        w_weights = np.pad(
            w_weights,
            ((0, 0), (0, spatial_embedding_dim - w_dim)),
            mode="constant",
        )

    # Set weights for spatial embeddings first
    keras_model.x_position_embeddings.set_weights([x_weights])
    keras_model.y_position_embeddings.set_weights([y_weights])
    keras_model.h_position_embeddings.set_weights([h_weights])
    keras_model.w_position_embeddings.set_weights([w_weights])

    # Create projection matrices based on actual weight shapes
    x_proj = np.random.normal(
        0, 0.02, (spatial_embedding_dim, hf_config.hidden_size)
    )
    y_proj = np.random.normal(
        0, 0.02, (spatial_embedding_dim, hf_config.hidden_size)
    )
    h_proj = np.random.normal(
        0, 0.02, (spatial_embedding_dim, hf_config.hidden_size)
    )
    w_proj = np.random.normal(
        0, 0.02, (spatial_embedding_dim, hf_config.hidden_size)
    )

    # Set weights for projection layers
    keras_model.x_proj.set_weights([x_proj, np.zeros(hf_config.hidden_size)])
    keras_model.y_proj.set_weights([y_proj, np.zeros(hf_config.hidden_size)])
    keras_model.h_proj.set_weights([h_proj, np.zeros(hf_config.hidden_size)])
    keras_model.w_proj.set_weights([w_proj, np.zeros(hf_config.hidden_size)])

    # Token type embeddings
    keras_model.token_type_embeddings.set_weights(
        [hf_weights["embeddings.token_type_embeddings.weight"].numpy()]
    )

    # Layer normalization
    keras_model.embeddings_LayerNorm.set_weights(
        [
            hf_weights["embeddings.LayerNorm.weight"].numpy(),
            hf_weights["embeddings.LayerNorm.bias"].numpy(),
        ]
    )

    # Transformer layers
    for i in range(hf_config.num_hidden_layers):
        # Attention
        keras_model.encoder_layers[i].attention.q_proj.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.attention.self.query.weight"]
                .numpy()
                .T,
                hf_weights[
                    f"encoder.layer.{i}.attention.self.query.bias"
                ].numpy(),
            ]
        )
        keras_model.encoder_layers[i].attention.k_proj.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.attention.self.key.weight"]
                .numpy()
                .T,
                hf_weights[
                    f"encoder.layer.{i}.attention.self.key.bias"
                ].numpy(),
            ]
        )
        keras_model.encoder_layers[i].attention.v_proj.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.attention.self.value.weight"]
                .numpy()
                .T,
                hf_weights[
                    f"encoder.layer.{i}.attention.self.value.bias"
                ].numpy(),
            ]
        )
        keras_model.encoder_layers[i].attention.out_proj.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.attention.output.dense.weight"]
                .numpy()
                .T,
                hf_weights[
                    f"encoder.layer.{i}.attention.output.dense.bias"
                ].numpy(),
            ]
        )

        # Attention output layer norm
        keras_model.encoder_layers[i].attention_output_layernorm.set_weights(
            [
                hf_weights[
                    f"encoder.layer.{i}.attention.output.LayerNorm.weight"
                ].numpy(),
                hf_weights[
                    f"encoder.layer.{i}.attention.output.LayerNorm.bias"
                ].numpy(),
            ]
        )

        # Intermediate
        keras_model.encoder_layers[i].intermediate_dense.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.intermediate.dense.weight"]
                .numpy()
                .T,
                hf_weights[
                    f"encoder.layer.{i}.intermediate.dense.bias"
                ].numpy(),
            ]
        )

        # Output
        keras_model.encoder_layers[i].output_dense.set_weights(
            [
                hf_weights[f"encoder.layer.{i}.output.dense.weight"].numpy().T,
                hf_weights[f"encoder.layer.{i}.output.dense.bias"].numpy(),
            ]
        )
        keras_model.encoder_layers[i].output_layernorm.set_weights(
            [
                hf_weights[
                    f"encoder.layer.{i}.output.LayerNorm.weight"
                ].numpy(),
                hf_weights[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy(),
            ]
        )

    # Final layer norm
    keras_model.norm.set_weights(
        [
            hf_weights["norm.weight"].numpy(),
            hf_weights["norm.bias"].numpy(),
        ]
    )

    # CLS token
    keras_model.cls_token.assign(hf_weights["cls_token"].numpy())

    # Patch embedding
    patch_embed_weight = hf_weights["patch_embed.proj.weight"].numpy()
    # Reshape to (height, width, in_channels, out_channels)
    patch_embed_weight = np.transpose(patch_embed_weight, (2, 3, 1, 0))
    keras_model.patch_embed.set_weights(
        [patch_embed_weight, hf_weights["patch_embed.proj.bias"].numpy()]
    )

    # Patch embedding layer norm
    keras_model.patch_embed_layer_norm.set_weights(
        [
            hf_weights["LayerNorm.weight"].numpy(),
            hf_weights["LayerNorm.bias"].numpy(),
        ]
    )

    # Save the model
    keras_model.save(os.path.join(output_dir, f"layoutlmv3_{model_size}.keras"))

    # Save the configuration
    config = {
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.hidden_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "intermediate_size": hf_config.intermediate_size,
        "hidden_act": hf_config.hidden_act,
        "hidden_dropout_prob": hf_config.hidden_dropout_prob,
        "attention_probs_dropout_prob": hf_config.attention_probs_dropout_prob,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "type_vocab_size": hf_config.type_vocab_size,
        "initializer_range": hf_config.initializer_range,
        "layer_norm_eps": hf_config.layer_norm_eps,
        "image_size": (112, 112),
        "patch_size": 16,
        "num_channels": 3,
        "qkv_bias": True,
        "use_abs_pos": True,
        "use_rel_pos": False,
        "rel_pos_bins": 32,
        "max_rel_pos": 128,
        "spatial_embedding_dim": spatial_embedding_dim,
    }

    with open(
        os.path.join(output_dir, f"layoutlmv3_{model_size}_config.json"), "w"
    ) as f:
        json.dump(config, f, indent=2)

    # Save the vocabulary
    vocab = hf_tokenizer.get_vocab()
    # Ensure special tokens are in the vocabulary
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Save vocabulary
    vocab_path = os.path.join(output_dir, f"layoutlmv3_{model_size}_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    # Save tokenizer config
    tokenizer_config = {
        "lowercase": True,
        "strip_accents": True,
        "oov_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
    }
    config_path = os.path.join(
        output_dir, f"layoutlmv3_{model_size}_tokenizer_config.json"
    )
    with open(config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\nSuccessfully converted {hf_model_name_or_path} to Keras format")
    print(f"Output saved to {output_dir}")


def main():
    """Convert LayoutLMv3 checkpoints."""
    # Convert base model
    convert_checkpoint(
        "microsoft/layoutlmv3-base",
        "checkpoints/layoutlmv3",
        model_size="base",
    )

    # Convert large model
    convert_checkpoint(
        "microsoft/layoutlmv3-large",
        "checkpoints/layoutlmv3",
        model_size="large",
    )


if __name__ == "__main__":
    main()
