import json
import os
import shutil
import warnings

import jax.numpy as jnp
import keras
import keras.ops as ops


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
    backend = keras.config.backend()
    backbone = keras_model.backbone
    hf_config = convert_to_hf_config(backbone)

    weights_dict = {}

    # Map token embedding
    token_embedding_layer = backbone.get_layer("token_embedding")
    weights_dict["model.embed_tokens.weight"] = token_embedding_layer.weights[0]

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")

        # Pre-attention normalization
        weights_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer.pre_attention_norm.weights[0]
        )

        # Attention query projection
        query_kernel = decoder_layer.attention.query_dense.weights[0]
        query_kernel = ops.transpose(query_kernel, axes=(1, 0, 2))
        query_kernel = ops.reshape(query_kernel, (-1, backbone.hidden_dim))
        query_kernel = ops.transpose(query_kernel)
        weights_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = query_kernel

        # Attention key projection
        key_kernel = decoder_layer.attention.key_dense.weights[0][0]
        weights_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            ops.transpose(key_kernel)
        )

        # Attention value projection
        value_kernel = decoder_layer.attention.value_dense.weights[0][0]
        weights_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            ops.transpose(value_kernel)
        )

        # Attention output projection
        out_kernel = decoder_layer.attention.output_dense.weights[0]
        out_kernel = ops.transpose(out_kernel, axes=(2, 0, 1))
        out_kernel = ops.reshape(out_kernel, (backbone.hidden_dim, -1))
        weights_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = out_kernel

        # Post-attention normalization
        weights_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer.pre_ffw_norm.weights[0]
        )

        # MLP gate projection
        gate_kernel = decoder_layer.gating_ffw.weights[0]
        weights_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )

        # MLP up projection
        up_kernel = decoder_layer.gating_ffw_2.weights[0]
        weights_dict[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )

        # MLP down projection
        down_kernel = decoder_layer.ffw_linear.weights[0]
        weights_dict[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )

    # Map final normalization
    weights_dict["model.norm.weight"] = backbone.get_layer(
        "final_normalization"
    ).weights[0]

    # Tie lm_head.weight to embedding weights
    weights_dict["lm_head.weight"] = token_embedding_layer.weights[0]

    # Save config
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)

    # Save weights based on backend
    weights_path = os.path.join(path, "model.safetensors")
    if backend == "torch":
        try:
            from safetensors.torch import save_file

            weights_dict_contiguous = {
                k: v.contiguous() for k, v in weights_dict.items()
            }
            save_file(weights_dict_contiguous, weights_path)
        except ImportError:
            raise ImportError("Install `safetensors.torch` for Torch backend.")
    elif backend == "tensorflow":
        try:
            from safetensors.tensorflow import save_file

            save_file(weights_dict, weights_path)
        except ImportError:
            raise ImportError(
                "Install `safetensors.tensorflow` for TensorFlow backend."
            )
    elif backend == "jax":
        try:
            from safetensors.flax import save_file

            weights_dict_contiguous = {
                k: jnp.ascontiguousarray(v) for k, v in weights_dict.items()
            }
            save_file(weights_dict_contiguous, weights_path)
        except ImportError:
            raise ImportError("Install `safetensors.flax` for JAX backend.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

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
