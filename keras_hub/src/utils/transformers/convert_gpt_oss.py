import numpy as np

from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = GptOssBackbone


def convert_backbone_config(transformers_config):
    """
    Converts a Hugging Face Transformers GPT-OSS configuration to a KerasHub
    GptOssBackbone configuration.
    """
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "num_experts": transformers_config["num_local_experts"],
        "top_k": transformers_config["num_experts_per_tok"],
        "rope_max_wavelength": transformers_config["rope_theta"],
        "rope_scaling_factor": transformers_config.get("rope_scaling", 1.0),
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "sliding_window": transformers_config["sliding_window"],
        "dropout": transformers_config.get("attention_dropout", 0.0),
        "use_bias": transformers_config.get("attention_bias", False),
    }


def convert_weights(backbone, loader, transformers_config):
    """
    Converts Hugging Face Transformers GPT-OSS model weights to KerasHub
    GptOssBackbone weights.
    """
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    def transpose_and_reshape(x, shape):
        # PyTorch nn.Linear weights are (out_features, in_features)
        # Keras Dense layer kernels are (in_features, out_features)
        # Transpose and then reshape to match Keras variable shape
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm (GptOssRMSNorm)
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention layers (GptOssAttention)
        ## Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        if backbone.use_bias:
            loader.port_weight(
                keras_variable=decoder_layer._self_attention_layer.query_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.q_proj.bias",
            )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        if backbone.use_bias:
            loader.port_weight(
                keras_variable=decoder_layer._self_attention_layer.key_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.k_proj.bias",
            )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        if backbone.use_bias:
            loader.port_weight(
                keras_variable=decoder_layer._self_attention_layer.value_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.v_proj.bias",
            )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        if backbone.use_bias:
            loader.port_weight(
                keras_variable=decoder_layer._self_attention_layer.output_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.o_proj.bias",
            )
        ## Sinks (unique to GptOssAttention)
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.sinks,
            hf_weight_key=f"model.layers.{i}.self_attn.sinks",
        )

        # MoE layers (GptOssMLP)
        # Router gate (GptOssTopKRouter)
        loader.port_weight(
            keras_variable=decoder_layer._sparse_moe_block._sparse_feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.router.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._sparse_moe_block._sparse_feedforward_gate_dense.bias,
            hf_weight_key=f"model.layers.{i}.mlp.router.bias",
        )

        # Batched experts (GptOssExperts)
        # PyTorch GptOssExperts parameters:
        #   - gate_up_proj (num_experts, hidden_size, 2 * expert_dim)
        #   - gate_up_proj_bias (num_experts, 2 * expert_dim)
        #   - down_proj (num_experts, expert_dim, hidden_size)
        #   - down_proj_bias (num_experts, hidden_size)

        # KerasHub GptOssExpertBank variables (assuming separate kernel/bias variables):
        #   - _expert_feedforward_gate_kernel (num_experts, hidden_dim, intermediate_dim)
        #   - _expert_feedforward_gate_bias (num_experts, intermediate_dim)
        #   - _expert_feedforward_intermediate_kernel (num_experts, hidden_dim, intermediate_dim)
        #   - _expert_feedforward_intermediate_bias (num_experts, intermediate_dim)
        #   - _expert_feedforward_output_kernel (num_experts, intermediate_dim, hidden_dim)
        #   - _expert_feedforward_output_bias (num_experts, hidden_dim)

        hf_gate_up_proj = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj"
        )
        hf_gate_up_proj_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj_bias"
        )
        hf_down_proj = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj"
        )
        hf_down_proj_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj_bias"
        )

        # Extract gate (w1) and intermediate (w3) kernels and biases from gate_up_proj
        # PyTorch gate_up_proj[:, :, ::2] corresponds to w1 (gate kernel)
        # PyTorch gate_up_proj[:, :, 1::2] corresponds to w3 (intermediate kernel)
        # PyTorch gate_up_proj_bias[:, ::2] corresponds to b1 (gate bias)
        # PyTorch gate_up_proj_bias[:, 1::2] corresponds to b3 (intermediate bias)

        # Kernels: PyTorch (num_experts, hidden_size, expert_dim) -> Keras (num_experts, hidden_dim, intermediate_dim)
        # No transpose needed as shapes match (num_experts, input_dim, output_dim)
        gate_kernels = hf_gate_up_proj[:, :, ::2]
        intermediate_kernels = hf_gate_up_proj[:, :, 1::2]
        output_kernels = hf_down_proj  # PyTorch (num_experts, expert_dim, hidden_size) -> Keras (num_experts, intermediate_dim, hidden_dim)

        # Biases: PyTorch (num_experts, expert_dim) -> Keras (num_experts, intermediate_dim)
        gate_biases = hf_gate_up_proj_bias[:, ::2]
        intermediate_biases = hf_gate_up_proj_bias[:, 1::2]
        output_biases = hf_down_proj_bias  # PyTorch (num_experts, hidden_size) -> Keras (num_experts, hidden_dim)

        # Assign batched weights to expert_bank variables
        expert_bank = decoder_layer._sparse_moe_block.expert_bank

        expert_bank._expert_feedforward_gate_kernel.assign(gate_kernels)
        expert_bank._expert_feedforward_gate_bias.assign(gate_biases)

        expert_bank._expert_feedforward_intermediate_kernel.assign(
            intermediate_kernels
        )
        expert_bank._expert_feedforward_intermediate_bias.assign(
            intermediate_biases
        )

        expert_bank._expert_feedforward_output_kernel.assign(output_kernels)
        expert_bank._expert_feedforward_output_bias.assign(output_biases)

        # Feedforward layernorm (GptOssRMSNorm)
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer (GptOssRMSNorm)
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    """
    Converts a Hugging Face Transformers GPT-OSS tokenizer to a KerasHub
    tokenizer.
    """
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
