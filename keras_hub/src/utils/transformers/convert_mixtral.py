import numpy as np

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = MixtralBackbone


def convert_backbone_config(transformers_config):
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
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "sliding_window": transformers_config["sliding_window"],
        "output_router_logits": transformers_config["output_router_logits"],
    }


def convert_weights(backbone, loader, transformers_config):
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
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention layers
        ## Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MoE layers
        # Router gate
        loader.port_weight(
            keras_variable=decoder_layer._sparse_moe_block._sparse_feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.block_sparse_moe.gate.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Batched experts: w1 (gate), w3 (intermediate), and w2 (output) weights
        gate_weights_list = []
        intermediate_weights_list = []
        output_weights_list = []
        for expert_idx in range(backbone.num_experts):
            # Load w1 (gate dense) for each expert
            w1 = loader.get_tensor(
                f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            )
            w1_transposed = np.transpose(w1, axes=(1, 0))
            gate_weights_list.append(w1_transposed)

            w3 = loader.get_tensor(
                f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w3.weight"
            )
            w3_transposed = np.transpose(w3, axes=(1, 0))
            intermediate_weights_list.append(w3_transposed)

            w2 = loader.get_tensor(
                f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            )
            w2_transposed = np.transpose(w2, axes=(1, 0))
            output_weights_list.append(w2_transposed)

        gate_batched = np.stack(gate_weights_list, axis=0)
        intermediate_batched = np.stack(intermediate_weights_list, axis=0)
        output_batched = np.stack(output_weights_list, axis=0)

        # Assign batched weights to expert_bank
        decoder_layer._sparse_moe_block.expert_bank._expert_feedforward_gate_dense.assign(
            gate_batched
        )
        decoder_layer._sparse_moe_block.expert_bank._expert_feedforward_intermediate_dense.assign(
            intermediate_batched
        )
        decoder_layer._sparse_moe_block.expert_bank._expert_feedforward_output_dense.assign(
            output_batched
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
