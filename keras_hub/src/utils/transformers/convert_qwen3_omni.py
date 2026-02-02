import numpy as np

from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3OmniBackbone


def convert_backbone_config(transformers_config):
    # Qwen3-Omni has nested config:
    # thinker_config.text_config contains the model params
    text_config = transformers_config.get("thinker_config", {}).get(
        "text_config", transformers_config
    )

    # Extract mrope_section from rope_scaling dict (not top-level)
    rope_scaling = text_config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    return {
        "vocabulary_size": text_config["vocab_size"],
        "hidden_dim": text_config["hidden_size"],
        "head_dim": text_config["head_dim"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "intermediate_dim": text_config["intermediate_size"],
        "moe_intermediate_dim": text_config["moe_intermediate_size"],
        "num_experts": text_config["num_experts"],
        "num_experts_per_tok": text_config["num_experts_per_tok"],
        "norm_topk_prob": text_config["norm_topk_prob"],
        "decoder_sparse_step": text_config["decoder_sparse_step"],
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "rope_max_wavelength": text_config["rope_theta"],
        "mrope_section": tuple(mrope_section),
        "sliding_window_size": text_config.get("sliding_window"),
        "router_aux_loss_coefficient": text_config["router_aux_loss_coef"],
        "mlp_only_layers": text_config.get("mlp_only_layers", []),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
    }


def convert_weights(backbone, loader, transformers_config):
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="../lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"layers.{i}.input_layernorm.weight",
        )

        # Attention layers

        ## Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense_layer_norm.scale,
            hf_weight_key=f"layers.{i}.self_attn.q_norm.weight",
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense_layer_norm.scale,
            hf_weight_key=f"layers.{i}.self_attn.k_norm.weight",
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP layers
        if (
            (i not in backbone.mlp_only_layers)
            and backbone.num_experts > 0
            and ((i + 1) % backbone.decoder_sparse_step == 0)
        ):
            # MoE layers
            loader.port_weight(
                keras_variable=decoder_layer.mlp._sparse_feedforward_gate_dense.kernel,
                hf_weight_key=f"layers.{i}.mlp.gate.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Batched experts: gate_up_proj and down_proj
            gate_up_proj_list = []
            down_proj_list = []
            for expert_idx in range(backbone.num_experts):
                # Load gate_proj and up_proj for each expert
                gate_proj = loader.get_tensor(
                    f"layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight"
                )
                up_proj = loader.get_tensor(
                    f"layers.{i}.mlp.experts.{expert_idx}.up_proj.weight"
                )
                # Transpose to (hidden_dim, intermediate_dim)
                gate_proj = np.transpose(gate_proj, axes=(1, 0))
                up_proj = np.transpose(up_proj, axes=(1, 0))
                # Concatenate gate_proj and up_proj along the last dimension
                gate_up_proj = np.concatenate([gate_proj, up_proj], axis=-1)
                gate_up_proj_list.append(gate_up_proj)

                # Load down_proj for each expert
                down_proj = loader.get_tensor(
                    f"layers.{i}.mlp.experts.{expert_idx}.down_proj.weight"
                )
                down_proj = np.transpose(
                    down_proj, axes=(1, 0)
                )  # (intermediate_dim, hidden_dim)
                down_proj_list.append(down_proj)

            # Stack the lists to create batched weights
            gate_up_proj_batched = np.stack(
                gate_up_proj_list, axis=0
            )  # (num_experts, hidden_dim, 2 * intermediate_dim)
            down_proj_batched = np.stack(
                down_proj_list, axis=0
            )  # (num_experts, intermediate_dim, hidden_dim)

            # Assign batched weights to expert_bank
            decoder_layer.mlp.expert_bank._expert_feedforward_gate_dense.assign(
                gate_up_proj_batched
            )
            decoder_layer.mlp.expert_bank._expert_feedforward_output_dense.assign(
                down_proj_batched
            )
        else:
            loader.port_weight(
                keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
                hf_weight_key=f"layers.{i}.mlp.up_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=decoder_layer._feedforward_output_dense.kernel,
                hf_weight_key=f"layers.{i}.mlp.down_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=decoder_layer._feedforward_gate_dense.kernel,
                hf_weight_key=f"layers.{i}.mlp.gate_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    # Qwen3-Omni uses separate vocab.json and merges.txt files
    # (unlike Qwen3-MoE which has tokenizer.json)

    # Load vocab from vocab.json (flat dict: {token: id})
    vocab = load_json(preset, "vocab.json")

    # Load merges from merges.txt (text file with merge rules)
    merges_file = get_file(preset, "merges.txt")
    with open(merges_file, "r") as f:
        merges = [line.strip() for line in f if line.strip()]

    # Load special tokens from tokenizer_config.json
    tokenizer_config = load_json(preset, "tokenizer_config.json")

    # Extract special tokens from added_tokens_decoder
    special_tokens = []
    if "added_tokens_decoder" in tokenizer_config:
        for token_id, token_info in tokenizer_config[
            "added_tokens_decoder"
        ].items():
            content = token_info.get("content", "")
            # Skip reserved placeholder tokens
            if not content.startswith("<|reserved_special_token_"):
                special_tokens.append(content)
                # Add to vocab if not already present
                if content not in vocab:
                    vocab[content] = int(token_id)

    kwargs.update(
        {
            "unsplittable_tokens": special_tokens,
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)
