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

import keras

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer


def convert_to_embedding_preset(
    source_preset: str,
    output_preset: str,
    pooling_intermediate_dim: int,
    embedding_dim: int,
):
    """
    Converts a standard causal Gemma3 preset to an Embedding Gemma preset.

    This function loads a pre-trained causal Gemma3 backbone, reconfigures it
    for bi-directional attention and adds a pooling head, transfers the original
    weights, and saves the result as a new Keras Hub preset.

    Args:
        source_preset (str): The path or name of source causal Gemma3 preset.
        output_preset (str): The path to save the new embedding model preset.
        pooling_intermediate_dim (int): The intermediate dimension for the
            pooling head's dense layer.
        embedding_dim (int): The final output dimension of sentence embedding.
    """
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

    embedding_model = Gemma3Backbone.from_config(config)

    transferred_layers = 0
    source_layer_names = {layer.name for layer in source_model.layers}

    for target_layer in embedding_model.layers:
        if target_layer.name in source_layer_names:
            source_layer = source_model.get_layer(name=target_layer.name)
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())
                transferred_layers += 1

    os.makedirs(output_preset, exist_ok=True)
    embedding_model.save_to_preset(output_preset)
    source_tokenizer.save_to_preset(output_preset)
    print(f"Embedding Gemma preset successfully saved to: '{output_preset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pre-trained causal Gemma3 model to "
        "Embedding Gemma model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_preset",
        type=str,
        required=True,
        help="Path or name of the source causal Gemma3 preset "
        "(e.g., 'gemma3_instruct_4b_text').",
    )
    parser.add_argument(
        "--output_preset",
        type=str,
        required=True,
        help="Path to save the new Embedding Gemma preset "
        "(e.g., 'embedding_gemma3_4b_en').",
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

    convert_to_embedding_preset(
        source_preset=args.source_preset,
        output_preset=args.output_preset,
        pooling_intermediate_dim=args.pooling_intermediate_dim,
        embedding_dim=args.embedding_dim,
    )
