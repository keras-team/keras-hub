"""
Script to convert LayoutLMv3 checkpoints from Hugging Face to Keras format.
"""

import argparse
import json
import os

import keras
import numpy as np
from transformers import LayoutLMv3Config, LayoutLMv3Model

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import (
    LayoutLMv3Backbone,
)
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import (
    LayoutLMv3Tokenizer,
)


def convert_checkpoint(model_name):
    print(f"✨ Converting {model_name}...")
    
    # Load HuggingFace model and config
    hf_model = LayoutLMv3Model.from_pretrained(model_name)
    hf_config = LayoutLMv3Config.from_pretrained(model_name)
    hf_weights = hf_model.state_dict()
    
    # Create KerasHub model
    keras_model = LayoutLMv3Backbone(
        vocabulary_size=hf_config.vocab_size,
        hidden_dim=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        max_sequence_length=hf_config.max_position_embeddings,
        dtype="float32",
    )
    
    # Build model with dummy inputs
    dummy_inputs = {
        "token_ids": keras.ops.ones((1, 8), dtype="int32"),
        "padding_mask": keras.ops.ones((1, 8), dtype="int32"),
        "bbox": keras.ops.ones((1, 8, 4), dtype="int32"),
    }
    keras_model(dummy_inputs)

    # Token embeddings
    token_embedding_weight = hf_weights["embeddings.word_embeddings.weight"].numpy()
    keras_model.token_embedding.embeddings.assign(token_embedding_weight)
    print(f"✅ Token embedding: {token_embedding_weight.shape}")

    # Position embeddings
    position_weight = hf_weights["embeddings.position_embeddings.weight"].numpy()
    keras_model.position_embedding.position_embeddings.assign(position_weight)
    print(f"✅ Position embedding: {position_weight.shape}")

    # Token type embeddings
    token_type_weight = hf_weights["embeddings.token_type_embeddings.weight"].numpy()
    keras_model.token_type_embedding.embeddings.assign(token_type_weight)
    print(f"✅ Token type embedding: {token_type_weight.shape}")

    # Spatial embeddings and projections
    spatial_coords = ['x', 'y', 'h', 'w']
    spatial_projections = {}
    
    for coord in spatial_coords:
        # Spatial embedding
        spatial_key = f"embeddings.{coord}_position_embeddings.weight"
        if spatial_key in hf_weights:
            spatial_weight = hf_weights[spatial_key].numpy()
            spatial_emb = getattr(keras_model, f"{coord}_position_embedding")
            spatial_emb.embeddings.assign(spatial_weight)
            print(f"✅ {coord} spatial embedding: {spatial_weight.shape}")
        
        # Spatial projection
        proj_key = f"embeddings.{coord}_position_projection"
        if f"{proj_key}.weight" in hf_weights:
            proj_weight = hf_weights[f"{proj_key}.weight"].numpy().T
            proj_bias = hf_weights[f"{proj_key}.bias"].numpy()
            projection_layer = getattr(keras_model, f"{coord}_projection")
            projection_layer.kernel.assign(proj_weight)
            projection_layer.bias.assign(proj_bias)
            print(f"✅ {coord} projection: {proj_weight.shape}")

    # Layer norm and dropout
    ln_weight = hf_weights["embeddings.LayerNorm.weight"].numpy()
    ln_bias = hf_weights["embeddings.LayerNorm.bias"].numpy()
    keras_model.embeddings_layer_norm.gamma.assign(ln_weight)
    keras_model.embeddings_layer_norm.beta.assign(ln_bias)
    print(f"✅ Embeddings LayerNorm: {ln_weight.shape}")

    # Transformer layers
    for i in range(hf_config.num_hidden_layers):
        hf_prefix = f"encoder.layer.{i}"
        keras_layer = keras_model.transformer_layers[i]
        
        # Self attention
        q_weight = hf_weights[f"{hf_prefix}.attention.self.query.weight"].numpy().T
        k_weight = hf_weights[f"{hf_prefix}.attention.self.key.weight"].numpy().T
        v_weight = hf_weights[f"{hf_prefix}.attention.self.value.weight"].numpy().T
        q_bias = hf_weights[f"{hf_prefix}.attention.self.query.bias"].numpy()
        k_bias = hf_weights[f"{hf_prefix}.attention.self.key.bias"].numpy()
        v_bias = hf_weights[f"{hf_prefix}.attention.self.value.bias"].numpy()
        
        keras_layer._self_attention_layer._query_dense.kernel.assign(q_weight)
        keras_layer._self_attention_layer._key_dense.kernel.assign(k_weight)
        keras_layer._self_attention_layer._value_dense.kernel.assign(v_weight)
        keras_layer._self_attention_layer._query_dense.bias.assign(q_bias)
        keras_layer._self_attention_layer._key_dense.bias.assign(k_bias)
        keras_layer._self_attention_layer._value_dense.bias.assign(v_bias)
        
        # Attention output
        attn_out_weight = hf_weights[f"{hf_prefix}.attention.output.dense.weight"].numpy().T
        attn_out_bias = hf_weights[f"{hf_prefix}.attention.output.dense.bias"].numpy()
        keras_layer._self_attention_layer._output_dense.kernel.assign(attn_out_weight)
        keras_layer._self_attention_layer._output_dense.bias.assign(attn_out_bias)
        
        # Attention layer norm
        attn_ln_weight = hf_weights[f"{hf_prefix}.attention.output.LayerNorm.weight"].numpy()
        attn_ln_bias = hf_weights[f"{hf_prefix}.attention.output.LayerNorm.bias"].numpy()
        keras_layer._self_attention_layernorm.gamma.assign(attn_ln_weight)
        keras_layer._self_attention_layernorm.beta.assign(attn_ln_bias)
        
        # Feed forward
        ff1_weight = hf_weights[f"{hf_prefix}.intermediate.dense.weight"].numpy().T
        ff1_bias = hf_weights[f"{hf_prefix}.intermediate.dense.bias"].numpy()
        keras_layer._feedforward_intermediate_dense.kernel.assign(ff1_weight)
        keras_layer._feedforward_intermediate_dense.bias.assign(ff1_bias)
        
        ff2_weight = hf_weights[f"{hf_prefix}.output.dense.weight"].numpy().T
        ff2_bias = hf_weights[f"{hf_prefix}.output.dense.bias"].numpy()
        keras_layer._feedforward_output_dense.kernel.assign(ff2_weight)
        keras_layer._feedforward_output_dense.bias.assign(ff2_bias)
        
        # Output layer norm
        out_ln_weight = hf_weights[f"{hf_prefix}.output.LayerNorm.weight"].numpy()
        out_ln_bias = hf_weights[f"{hf_prefix}.output.LayerNorm.bias"].numpy()
        keras_layer._feedforward_layernorm.gamma.assign(out_ln_weight)
        keras_layer._feedforward_layernorm.beta.assign(out_ln_bias)
        
        print(f"✅ Transformer layer {i}")

    # Save the model
    preset_dir = f"layoutlmv3_{model_name.split('/')[-1]}_keras"
    os.makedirs(preset_dir, exist_ok=True)
    
    keras_model.save_preset(preset_dir)
    
    # Create tokenizer and save
    tokenizer = LayoutLMv3Tokenizer(
        vocabulary=os.path.join(preset_dir, "vocabulary.json"),
        merges=os.path.join(preset_dir, "merges.txt"),
    )
    tokenizer.save_preset(preset_dir)
    
    print(f"✅ Saved preset to {preset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default="microsoft/layoutlmv3-base",
        help="HuggingFace model name"
    )
    
    args = parser.parse_args()
    convert_checkpoint(args.model_name)


if __name__ == "__main__":
    main()
