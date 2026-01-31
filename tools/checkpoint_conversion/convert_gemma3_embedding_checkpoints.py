"""
Convert a pre-trained causal Gemma3 Keras model to an Embedding Gemma model.

This script takes a standard Gemma3 model designed for causal language modeling
and adapts it for sentence embedding tasks. It modifies the model architecture
for bi-directional attention, adds a new pooling head for generating fixed-size
embeddings, and transfers the weights from the original model.

Setup:
```shell
# Make sure to install the necessary libraries, including the specific
# keras_hub package containing the Gemma3 models.
pip install keras-hub
pip install keras

Usage:
```shell
cd tools/checkpoint_conversion
python3 convert_embedding_gemma_checkpoints.py \
    --source_preset gemma3_instruct_4b_text \
    --output_preset embedding_gemma3_4b_en \
    --pooling_intermediate_dim 4096 \
    --embedding_dim 768
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import kagglehub
import keras
import numpy as np
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer

PRESET_MAP = {
    "gemma3_embedding_270m_en": "gemma3_270m",
    "gemma3_embedding_instruct_270m_en": "gemma3_instruct_270m",
    "gemma3_embedding_1b_en": "gemma3_1b",
    "gemma3_embedding_instruct_1b_en": "gemma3_instruct_1b",
    "gemma3_embedding_4b_text_en": "gemma3_4b_text",
    "gemma3_embedding_instruct_4b_text_en": "gemma3_instruct_4b_text",
    "gemma3_embedding_12b_text_en": "gemma3_12b_text",
    "gemma3_embedding_instruct_12b_text_en": "gemma3_instruct_12b_text",
    "gemma3_embedding_27b_text_en": "gemma3_27b_text",
    "gemma3_embedding_instruct_27b_text_en": "gemma3_instruct_27b_text",
}


def validate_output(
    source_model,
    embedding_model,
):
    """
    Validates the output of the converted model against the source model.
    """
    print("Validating the converted model...")
    # Create a temporary validation model with the same weights as the
    # embedding model, but with causal attention and no pooling head.
    config = embedding_model.get_config()
    config["is_embedding_model"] = False
    config["use_bidirectional_attention"] = False

    # Instantiate validation model from config
    validation_model = Gemma3Backbone.from_config(config)

    # First, test that the number of parameters match
    source_params = source_model.count_params()
    validation_params = validation_model.count_params()
    print(f"Source model parameters: {source_params}")
    print(f"Validation model parameters: {validation_params}")
    if source_params == validation_params:
        print("✅ Parameter count match.")
    else:
        print("❌ Parameter count mismatch.")
    assert source_params == validation_params

    # Transfer weights from embedding_model to validation_model
    source_layer_names = {layer.name for layer in embedding_model.layers}
    transferred_layers = 0
    for target_layer in validation_model.layers:
        if target_layer.name in source_layer_names:
            source_layer = embedding_model.get_layer(name=target_layer.name)
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())
                transferred_layers += 1

    # Simple numerical verification
    inputs = {
        "token_ids": np.random.randint(0, 1000, (1, 16)),
        "padding_mask": np.ones((1, 16)),
    }

    print("Comparing outputs...")
    # Run forward pass
    # Source model (causal)
    output_source = ops.convert_to_numpy(source_model(inputs))
    # Validation model (converted, but running in causal mode)
    output_target = ops.convert_to_numpy(validation_model(inputs))

    # Compare
    diff = np.max(np.abs(output_source - output_target))
    print(f"Max Difference: {diff}")

    if diff < 1e-5:
        print("✅ VERIFICATION PASSED")
    else:
        print("❌ VERIFICATION FAILED")
        raise ValueError("Model verification failed! Outputs do not match.")


def convert_to_embedding_preset(
    source_preset,
    output_preset,
    pooling_intermediate_dim,
    embedding_dim,
):
    """
    Converts a standard causal Gemma3 preset to an Embedding Gemma preset.

    This function loads a pre-trained causal Gemma3 backbone, reconfigures it
    for bi-directional attention and adds a pooling head, transfers the original
    weights, and saves the result as a new Keras Hub preset.

    Args:
        source_preset: The path or name of source causal Gemma3 preset.
        output_preset: The path to save the new embedding model preset.
        pooling_intermediate_dim: The intermediate dimension for the
            pooling head's dense layer.
        embedding_dim: The final output dimension of sentence embedding.
    """
    print(f"Loading source model: {source_preset}...")
    source_model = Gemma3Backbone.from_preset(source_preset)
    source_tokenizer = Gemma3Tokenizer.from_preset(source_preset)

    config = source_model.get_config()

    config["is_embedding_model"] = True
    config["use_bidirectional_attention"] = True
    config["pooling_intermediate_dim"] = pooling_intermediate_dim
    config["embedding_dim"] = embedding_dim

    if config.get("vision_encoder") is not None:
        config["vision_encoder"] = keras.layers.deserialize(
            config["vision_encoder"]
        )

    print("Creating embedding model...")
    embedding_model = Gemma3Backbone.from_config(config)

    transferred_layers = 0
    source_layer_names = {layer.name for layer in source_model.layers}

    print("Transferring weights...")
    for target_layer in embedding_model.layers:
        if target_layer.name in source_layer_names:
            source_layer = source_model.get_layer(name=target_layer.name)
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())
                transferred_layers += 1

    # Validate output
    validate_output(source_model, embedding_model)

    print(f"Saving to preset: {output_preset}...")
    os.makedirs(output_preset, exist_ok=True)
    embedding_model.save_to_preset(output_preset)
    source_tokenizer.save_to_preset(output_preset)
    print(f"Embedding Gemma preset successfully saved to: '{output_preset}'")


if __name__ == "__main__":
    kagglehub.login()
    parser = argparse.ArgumentParser(
        description="Convert a pre-trained causal Gemma3 model to "
        "Embedding Gemma model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Name of the preset to convert. Must be one of "
        f"{list(PRESET_MAP.keys())}.",
    )
    parser.add_argument(
        "--source_preset",
        type=str,
        help="Path or name of the source causal Gemma3 preset "
        "(e.g., 'gemma3_instruct_4b_text'). Required if --preset is not set.",
    )
    parser.add_argument(
        "--output_preset",
        type=str,
        help="Path to save the new Embedding Gemma preset "
        "(e.g., 'embedding_gemma3_4b_en'). Required if --preset is not set.",
    )
    parser.add_argument(
        "--pooling_intermediate_dim",
        type=int,
        default=4096,
        help="Intermediate dimension for the pooling head's first dense layer.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="The final output dimension of the embedding projection.",
    )

    args = parser.parse_args()

    if args.preset:
        if args.preset not in PRESET_MAP:
            raise ValueError(
                f"Invalid preset {args.preset}. Must be one of "
                f"{list(PRESET_MAP.keys())}."
            )
        source_preset = PRESET_MAP[args.preset]
        output_preset = args.preset
    else:
        if not args.source_preset or not args.output_preset:
            parser.error(
                "Both --source_preset and --output_preset are required if "
                "--preset is not provided."
            )
        source_preset = args.source_preset
        output_preset = args.output_preset

    convert_to_embedding_preset(
        source_preset=source_preset,
        output_preset=output_preset,
        pooling_intermediate_dim=args.pooling_intermediate_dim,
        embedding_dim=args.embedding_dim,
    )
