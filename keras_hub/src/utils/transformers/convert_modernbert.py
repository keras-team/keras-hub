import torch
import numpy as np
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone

def convert_weights(pytorch_model_path, keras_model):
    # Load PyTorch state dict
    state_dict = torch.load(pytorch_model_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

   
    # Embeddings
    keras_model.get_layer("token_embedding").set_weights([
        state_dict["model.embeddings.tok_embeddings.weight"].numpy()
    ])

    # Encoder Layers
    for i in range(keras_model.num_layers):
        pt_prefix = f"model.layers.{i}"
        keras_layer = keras_model.get_layer(f"transformer_layer_{i}")
        
        # Norms (ModernBERT uses RMSNorm, usually only has 'weight')
        keras_layer.attn_norm.set_weights([
            state_dict[f"{pt_prefix}.attn_norm.weight"].numpy()
        ])
        keras_layer.mlp_norm.set_weights([
            state_dict[f"{pt_prefix}.mlp_norm.weight"].numpy()
        ])

        # Attention QKV (PyTorch packs these into one matrix)
        qkv_weight = state_dict[f"{pt_prefix}.attn.Wqkv.weight"].numpy()
        # Keras Dense layers expect (input_dim, output_dim)
        keras_layer.attn._qkv_dense.set_weights([qkv_weight.T]) 
        
        # Attention Output
        out_weight = state_dict[f"{pt_prefix}.attn.out_proj.weight"].numpy()
        keras_layer.attn._output_dense.set_weights([out_weight.T])

        # MLP (ModernBERT uses GeGLU: wi_0 and wi_1)
        wi_weight = state_dict[f"{pt_prefix}.mlp.wi.weight"].numpy()
        keras_layer.mlp._wi_dense.set_weights([wi_weight.T])
        
        wo_weight = state_dict[f"{pt_prefix}.mlp.wo.weight"].numpy()
        keras_layer.mlp._wo_dense.set_weights([wo_weight.T])

    print("Weights converted successfully!")

# Usage:
# backbone = ModernBertBackbone.from_preset("modern_bert_base", load_weights=False)
# convert_weights("pytorch_model.bin", backbone)
# backbone.save_weights("modern_bert_base.weights.h5")