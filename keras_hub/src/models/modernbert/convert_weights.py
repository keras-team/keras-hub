import os
import torch
import numpy as np
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone

def convert_modern_bert_weights(pt_weights_path, preset_dir):
    # Initialize a base model with the architecture you want to convert
    # For ModernBERT-base: 22 layers, 768 hidden_dim
    model = ModernBertBackbone(
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    )

    # Load weights
    state_dict = torch.load(pt_weights_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Map weights layer by layer
    # Token Embeddings
    model.get_layer("token_embedding").set_weights([
        state_dict["model.embeddings.tok_embeddings.weight"].numpy()
    ])

    for i in range(model.num_layers):
        pt_prefix = f"model.layers.{i}"
        # Make sure your backbone uses these names during layer creation
        keras_layer = model.get_layer(f"transformer_layer_{i}")

        # Norms (RMSNorm has no bias)
        keras_layer.attn_norm.set_weights([state_dict[f"{pt_prefix}.attn_norm.weight"].numpy()])
        keras_layer.mlp_norm.set_weights([state_dict[f"{pt_prefix}.mlp_norm.weight"].numpy()])

        # Attention QKV - PyTorch (3*hidden, hidden) -> Keras (hidden, 3*hidden)
        qkv_w = state_dict[f"{pt_prefix}.attn.Wqkv.weight"].numpy().T
        keras_layer.attn._qkv_dense.set_weights([qkv_w])

        # Attention Output
        out_w = state_dict[f"{pt_prefix}.attn.out_proj.weight"].numpy().T
        keras_layer.attn._output_dense.set_weights([out_w])

        # MLP Gated Linear Unit
        wi_w = state_dict[f"{pt_prefix}.mlp.wi.weight"].numpy().T
        keras_layer.mlp._wi_dense.set_weights([wi_w])
        
        wo_w = state_dict[f"{pt_prefix}.mlp.wo.weight"].numpy().T
        keras_layer.mlp._wo_dense.set_weights([wo_w])

    # Save to preset directory
    # This creates model.weights.h5 and config.json automatically
    model.save_to_preset(preset_dir)
    print(f"Preset saved successfully to: {preset_dir}")

# To run:
# convert_modern_bert_weights("path/to/modernbert-base.pth", "./modern_bert_base_preset")