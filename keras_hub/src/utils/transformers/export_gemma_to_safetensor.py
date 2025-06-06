import os
import torch
import numpy as np
import transformers
import keras_hub
import shutil
import json

# Set the Keras backend to torch/jax/tensorflow
os.environ["KERAS_BACKEND"] = "jax"
print(f"Using Keras backend: {os.environ['KERAS_BACKEND']}")

def convert_to_hf_config(keras_config):
    """Convert Keras Gemma config to Hugging Face GemmaConfig."""
    hf_config = transformers.GemmaConfig(
        vocab_size=keras_config.vocabulary_size,
        num_hidden_layers=keras_config.num_layers,
        num_attention_heads=keras_config.num_query_heads,
        num_key_value_heads=keras_config.num_key_value_heads,
        hidden_size=keras_config.hidden_dim,
        intermediate_size=keras_config.intermediate_dim // 2,
        head_dim=keras_config.head_dim,
        max_position_embeddings=8192,
    )
    return hf_config

def export_to_hf(keras_model, path, preset="gemma_2b_en"):
    """Convert a Keras Gemma model to Hugging Face format and save to path."""
    backbone = keras_model.backbone
    hf_config = convert_to_hf_config(backbone)
    hf_model = transformers.GemmaForCausalLM(hf_config)

    weights_dict = {}

    # Map token embedding
    token_embedding = backbone.get_layer("token_embedding").get_weights()[0]  # Shape: [vocab_size, hidden_size]
    weights_dict['model.embed_tokens.weight'] = torch.from_numpy(token_embedding)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")

        # Pre-attention normalization (input_layernorm)
        pre_attn_norm = decoder_layer.pre_attention_norm.get_weights()[0]  # Shape: [hidden_size]
        weights_dict[f'model.layers.{i}.input_layernorm.weight'] = torch.from_numpy(pre_attn_norm)

        # Attention query projection (q_proj)
        query_kernel = decoder_layer.attention.query_dense.get_weights()[0]  # Shape: [hidden_size, num_heads, head_dim]
        query_kernel = torch.from_numpy(query_kernel).permute(1, 0, 2).reshape(-1, backbone.hidden_dim).T  # Reshape to [hidden_size, num_heads * head_dim]
        weights_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = query_kernel

        # Attention key projection (k_proj)
        key_kernel = decoder_layer.attention.key_dense.get_weights()[0][0]  # Shape: [hidden_size, head_dim]
        key_kernel = torch.from_numpy(key_kernel).T  # Shape: [head_dim, hidden_size]
        weights_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = key_kernel

        # Attention value projection (v_proj)
        value_kernel = decoder_layer.attention.value_dense.get_weights()[0][0]  # Shape: [hidden_size, head_dim]
        value_kernel = torch.from_numpy(value_kernel).T  # Shape: [head_dim, hidden_size]
        weights_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = value_kernel

        # Attention output projection (o_proj)
        out_kernel = decoder_layer.attention.output_dense.get_weights()[0]  # Shape: [num_heads, head_dim, hidden_size]
        out_kernel = torch.from_numpy(out_kernel).permute(2, 0, 1).reshape(backbone.hidden_dim, -1)  # Reshape to [hidden_size, num_heads * head_dim]
        weights_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = out_kernel

        # Post-attention normalization (post_attention_layernorm)
        post_attn_norm = decoder_layer.pre_ffw_norm.get_weights()[0]  # Shape: [hidden_size]
        weights_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.from_numpy(post_attn_norm)

        # MLP gate projection
        gate_kernel = decoder_layer.gating_ffw.get_weights()[0]  # Shape: [hidden_size, intermediate_size]
        gate_kernel = torch.from_numpy(gate_kernel).T  # Shape: [intermediate_size, hidden_size]
        weights_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = gate_kernel

        # MLP up projection
        up_kernel = decoder_layer.gating_ffw_2.get_weights()[0]  # Shape: [hidden_size, intermediate_size]
        up_kernel = torch.from_numpy(up_kernel).T  # Shape: [intermediate_size, hidden_size]
        weights_dict[f'model.layers.{i}.mlp.up_proj.weight'] = up_kernel

        # MLP down projection
        down_kernel = decoder_layer.ffw_linear.get_weights()[0]  # Shape: [intermediate_size, hidden_size]
        down_kernel = torch.from_numpy(down_kernel).T  # Shape: [hidden_size, intermediate_size]
        weights_dict[f'model.layers.{i}.mlp.down_proj.weight'] = down_kernel

    # Map final normalization
    final_norm = backbone.get_layer("final_normalization").get_weights()[0]  # Shape: [hidden_size]
    weights_dict['model.norm.weight'] = torch.from_numpy(final_norm)

    # Tie lm_head.weight to embedding weights
    weights_dict['lm_head.weight'] = weights_dict['model.embed_tokens.weight']

    # Load weights into Hugging Face model
    print("Loading weights into Hugging Face model...")
    hf_model.load_state_dict(weights_dict, strict=False)

    # Save tokenizer
    os.makedirs(path, exist_ok=True)
    proto = keras_model.preprocessor.tokenizer.proto
    tokenizer_model_path = os.path.join(path, "tokenizer.model")
    if isinstance(proto, str):
        shutil.copyfile(proto, tokenizer_model_path)
    elif isinstance(proto, bytes):
        with open(tokenizer_model_path, "wb") as f:
            f.write(proto)
    else:
        raise ValueError("Unexpected type for tokenizer.proto")

    # Initialize and save Hugging Face tokenizer
    hf_tokenizer = transformers.GemmaTokenizer(tokenizer_model_path)
    hf_tokenizer.save_pretrained(path)

    # Save model
    hf_model.save_pretrained(path, safe_serialization=True)
    print(f"Model and tokenizer saved to {path}")

    return keras_model, hf_model, hf_tokenizer

if __name__ == "__main__":
    # Load Keras model
    keras_model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    input_text = "RCB is the new champ."
    max_length = 50

    # Export to Hugging Face format
    keras_model, hf_model, hf_tokenizer = export_to_hf(keras_model, "./export_to_hf")

    # Generate text with Keras model
    print("\nGenerating text with Keras model...")
    keras_output = keras_model.generate(input_text, max_length=max_length)
    print("Keras Output:", keras_output)

    # Generate text with Hugging Face model
    print("\nGenerating text with Hugging Face model...")
    hf_inputs = hf_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        hf_outputs = hf_model.generate(**hf_inputs, max_length=max_length, do_sample=False)
    hf_output_text = hf_tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print("Hugging Face Output:", hf_output_text)

    # Compare outputs
    if keras_output == hf_output_text:
        print("Outputs are identical.")
    else:
        print("Outputs differ.")