import json
import os
import shutil
import warnings

import torch
from safetensors.torch import save_file


def convert_to_hf_config(keras_config):
    hf_config = {
        "vocab_size": keras_config.vocabulary_size,
        "num_hidden_layers": keras_config.num_layers,
        "num_attention_heads": keras_config.num_query_heads,
        "num_key_value_heads": keras_config.num_key_value_heads,
        "hidden_size": keras_config.hidden_dim,
        "intermediate_size": keras_config.intermediate_dim // 2,
        "head_dim": keras_config.head_dim,
        "max_position_embeddings": 8192,
    }
    return hf_config


def export_to_hf(keras_model, path):
    """This function converts a Keras Gemma model to Hugging Face format by:
    - Extracting and mapping weights from the Keras backbone to safetensors.
    - Saving the configuration as 'config.json'.
    - Saving weights in 'model.safetensors'.
    - Saving tokenizer assets.
    Args:
        keras_model: The Keras Gemma model (e.g., GemmaCausalLM) to convert.
        path: str. Path of the directory to which the safetensors file,
        config and tokenizer will be saved.
    """
    backbone = keras_model.backbone
    hf_config = convert_to_hf_config(backbone)

    weights_dict = {}

    # Map token embedding
    token_embedding = backbone.get_layer("token_embedding").get_weights()[0]
    weights_dict["model.embed_tokens.weight"] = torch.from_numpy(
        token_embedding
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")

        # Pre-attention normalization
        pre_attn_norm = decoder_layer.pre_attention_norm.get_weights()[0]
        weights_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            torch.from_numpy(pre_attn_norm)
        )

        # Attention query projection
        query_kernel = decoder_layer.attention.query_dense.get_weights()[0]
        query_kernel = (
            torch.from_numpy(query_kernel)
            .permute(1, 0, 2)
            .reshape(-1, backbone.hidden_dim)
            .T
        )
        weights_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = query_kernel

        # Attention key projection
        key_kernel = decoder_layer.attention.key_dense.get_weights()[0][0]
        key_kernel = torch.from_numpy(key_kernel).T
        weights_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = key_kernel

        # Attention value projection
        value_kernel = decoder_layer.attention.value_dense.get_weights()[0][0]
        value_kernel = torch.from_numpy(value_kernel).T
        weights_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = value_kernel

        # Attention output projection
        out_kernel = decoder_layer.attention.output_dense.get_weights()[0]
        out_kernel = (
            torch.from_numpy(out_kernel)
            .permute(2, 0, 1)
            .reshape(backbone.hidden_dim, -1)
        )
        weights_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = out_kernel

        # Post-attention normalization
        post_attn_norm = decoder_layer.pre_ffw_norm.get_weights()[0]
        weights_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            torch.from_numpy(post_attn_norm)
        )

        # MLP gate projection
        gate_kernel = decoder_layer.gating_ffw.get_weights()[0]
        gate_kernel = torch.from_numpy(gate_kernel).T
        weights_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_kernel

        # MLP up projection
        up_kernel = decoder_layer.gating_ffw_2.get_weights()[0]
        up_kernel = torch.from_numpy(up_kernel).T
        weights_dict[f"model.layers.{i}.mlp.up_proj.weight"] = up_kernel

        # MLP down projection
        down_kernel = decoder_layer.ffw_linear.get_weights()[0]
        down_kernel = torch.from_numpy(down_kernel).T
        weights_dict[f"model.layers.{i}.mlp.down_proj.weight"] = down_kernel

    # Map final normalization
    final_norm = backbone.get_layer("final_normalization").get_weights()[0]
    weights_dict["model.norm.weight"] = torch.from_numpy(final_norm)

    # Tie lm_head.weight to embedding weights
    weights_dict["lm_head.weight"] = weights_dict[
        "model.embed_tokens.weight"
    ].clone()

    # Save config
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)

    # Make tensors contiguous before saving
    weights_dict_contiguous = {
        k: v.contiguous() for k, v in weights_dict.items()
    }

    # Save weights
    weights_path = os.path.join(path, "model.safetensors")
    save_file(weights_dict_contiguous, weights_path)

    # Save tokenizer assets
    keras_model.preprocessor.tokenizer.save_assets(path)

    # Rename vocabulary file
    vocab_spm_path = os.path.join(path, "vocabulary.spm")
    tokenizer_model_path = os.path.join(path, "tokenizer.model")
    if os.path.exists(vocab_spm_path):
        shutil.move(vocab_spm_path, tokenizer_model_path)
    else:
        warnings.warn(
            f"{vocab_spm_path} not found. Tokenizer may not load correctly."
        )
