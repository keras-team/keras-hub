import torch
import numpy as np
from modernbert_backbone import ModernBertBackbone

def convert_weights(hf_model_id="answerdotai/ModernBERT-base"):
    # Load PyTorch weights from Hugging Face
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(hf_model_id)
    state_dict = hf_model.state_dict()

    # Initialize your Keras Backbone with matching config
    # Adjust these dims based on whether you are using 'base' or 'large'
    keras_model = ModernBertBackbone(
        vocabulary_size=50368,
        num_layers=22,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=1152,
        local_attention_window=128,
    )

    # Mapping Function
    def set_keras_weights(layer_name, torch_weight_key, transpose=False):
        w = state_dict[torch_weight_key].numpy()
        if transpose:
            w = w.T # Linear layers in PyTorch are (out, in), Keras are (in, out)
        keras_model.get_layer(layer_name).set_weights([w])

    # Token Embeddings
    # ModernBERT uses 'model.embeddings.tok_embeddings.weight'
    set_keras_weights("token_embedding", "model.embeddings.token_embeddings.weight")
    set_keras_weights("token_embedding", "model.embeddings.tok_embeddings.weight")
    
    # Encoder Layers
    for i in range(22):
        # ModernBERT keys usually start with 'model.layers.i.'
        prefix = f"model.layers.{i}."
        k_prefix = f"transformer_layer_{i}_"
        
        # QKV Attention
        # ModernBERT uses a single fused Wqkv layer. 
        # If your Keras model has separate layers, you'll need to split this weight.
        set_keras_weights(f"{k_prefix}query_dense", f"{prefix}attn.Wqkv.weight", transpose=True)
        set_keras_weights(f"{k_prefix}output_dense", f"{prefix}attn.Wo.weight", transpose=True)
        
        # MLP (ModernBERT uses GeGLU, usually 'wi' and 'wo')
        set_keras_weights(f"{k_prefix}intermediate_dense", f"{prefix}mlp.wi.weight", transpose=True)
        set_keras_weights(f"{k_prefix}output_mlp_dense", f"{prefix}mlp.wo.weight", transpose=True)
        
        # Norms (ModernBERT uses 'attn_norm' and 'mlp_norm')
        set_keras_weights(f"{k_prefix}pre_attention_norm", f"{prefix}attn_norm.weight")
        set_keras_weights(f"{k_prefix}pre_mlp_norm", f"{prefix}mlp_norm.weight")
        
        set_keras_weights("final_norm", "model.final_norm.weight")
        set_keras_weights("token_embedding", "embeddings.word_embeddings.weight")

    # 4. Save to Keras .h5 or .weights.h5
    keras_model.save_weights("modernbert_base.weights.h5")
    print("Weights converted and saved successfully!")

convert_weights()