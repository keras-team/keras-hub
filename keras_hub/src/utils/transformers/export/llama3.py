import keras.ops as ops


def get_llama3_config(backbone):
    """Convert Keras Llama3 backbone config to Hugging Face dictionary.

    Args:
        backbone: Llama3Backbone. The Keras Llama3 backbone instance.

    Returns:
        A dict containing the Hugging Face model configuration.
    """
    head_dim = backbone.hidden_dim // backbone.num_query_heads

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "rope_theta": backbone.rope_max_wavelength,
        "hidden_act": "silu",
        "attention_dropout": backbone.dropout,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "use_cache": True,
        "torch_dtype": backbone.dtype_policy.compute_dtype,
    }

    # Llama 3.1+ uses scaled RoPE ("llama3" rope_type).
    # Use getattr with None defaults so that older Llama 3.0 backbones that
    # lack these attributes do not raise AttributeError.
    rope_freq_adj = getattr(backbone, "rope_frequency_adjustment_factor", None)
    if rope_freq_adj is not None:
        hf_config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": rope_freq_adj,
            "low_freq_factor": getattr(backbone, "rope_low_freq_factor", None),
            "high_freq_factor": getattr(
                backbone, "rope_high_freq_factor", None
            ),
            "original_max_position_embeddings": getattr(
                backbone, "rope_pretraining_sequence_length", None
            ),
        }

    return hf_config


def get_llama3_weights_map(backbone, include_lm_head=False):
    """Create a Keras-to-HuggingFace weight name mapping for Llama3.

    Args:
        backbone: Llama3Backbone. A `keras_hub.models.Llama3Backbone` instance.
        include_lm_head: bool. If True, includes the ``lm_head.weight`` tensor
            (used when exporting a CausalLM task).

    Returns:
        A dict mapping HuggingFace weight keys to Keras tensors.
    """
    weights_map = {}

    # Token embeddings
    token_embedding_layer = backbone.get_layer("token_embedding")
    weights_map["model.embed_tokens.weight"] = token_embedding_layer.embeddings

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        attn_layer = decoder_layer._self_attention_layer

        # --- Normalization ---
        # Pre-attention (input) layernorm
        weights_map[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer._self_attention_layernorm.scale
        )
        # Pre-MLP (post-attention) layernorm
        weights_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer._feedforward_layernorm.scale
        )

        # --- Attention projections ---
        # Keras Q kernel: (hidden_dim, num_query_heads, head_dim)
        # HF   q_proj.weight: (num_query_heads * head_dim, hidden_dim)
        q_kernel = attn_layer._query_dense.kernel
        q_kernel = ops.reshape(q_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            ops.transpose(q_kernel)
        )

        # Keras K kernel: (hidden_dim, num_key_value_heads, head_dim)
        # HF   k_proj.weight: (num_key_value_heads * head_dim, hidden_dim)
        k_kernel = attn_layer._key_dense.kernel
        k_kernel = ops.reshape(k_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            ops.transpose(k_kernel)
        )

        # Keras V kernel: (hidden_dim, num_key_value_heads, head_dim)
        # HF   v_proj.weight: (num_key_value_heads * head_dim, hidden_dim)
        v_kernel = attn_layer._value_dense.kernel
        v_kernel = ops.reshape(v_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            ops.transpose(v_kernel)
        )

        # Keras O kernel: (num_query_heads, head_dim, hidden_dim)
        # HF   o_proj.weight: (hidden_dim, num_query_heads * head_dim)
        o_kernel = attn_layer._output_dense.kernel
        o_kernel = ops.reshape(o_kernel, (-1, backbone.hidden_dim))
        weights_map[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            ops.transpose(o_kernel)
        )

        # --- MLP (SwiGLU) ---
        # Keras gate/up kernel: (hidden_dim, intermediate_dim)
        # HF   gate/up_proj.weight: (intermediate_dim, hidden_dim)
        gate_kernel = decoder_layer._feedforward_gate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )

        up_kernel = decoder_layer._feedforward_intermediate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )

        # Keras down kernel: (intermediate_dim, hidden_dim)
        # HF   down_proj.weight: (hidden_dim, intermediate_dim)
        down_kernel = decoder_layer._feedforward_output_dense.kernel
        weights_map[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )

    # Final layernorm
    weights_map["model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM head (only when exporting the full CausalLM task)
    if include_lm_head:
        if backbone.tie_word_embeddings:
            weights_map["lm_head.weight"] = weights_map[
                "model.embed_tokens.weight"
            ]
        else:
            weights_map["lm_head.weight"] = ops.transpose(
                token_embedding_layer.reverse_embeddings
            )

    return weights_map


def get_llama3_tokenizer_config(tokenizer):
    """Build a HuggingFace-compatible tokenizer_config.json for Llama3.

    Args:
        tokenizer: Llama3Tokenizer. The Keras Llama3 tokenizer instance.

    Returns:
        A dict containing the Hugging Face tokenizer configuration.
    """
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": tokenizer.start_token,
        "eos_token": tokenizer.end_token,
        "pad_token": None,
        "unk_token": None,
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 131072,
    }
    return tokenizer_config


def build_llama3_tokenizer_json(tokenizer):
    """Build the tokenizer.json needed by PreTrainedTokenizerFast.

    HuggingFace's fast tokenizer requires a ``tokenizer.json`` in the
    tokenizers-library format.  This function constructs that JSON dict
    directly from the vocab and merges held by the KerasHub tokenizer,
    so no extra dependency beyond the standard ``json`` module is needed.

    Args:
        tokenizer: Llama3Tokenizer. A `keras_hub.models.Llama3Tokenizer`
            instance.

    Returns:
        A dict suitable for serialising as ``tokenizer.json``.
    """
    # Support both dict {str: int} and list-of-strings vocabulary formats.
    raw_vocab = tokenizer.vocabulary
    if isinstance(raw_vocab, dict):
        vocab = dict(raw_vocab)  # {str: int}
    else:
        vocab = {token: i for i, token in enumerate(raw_vocab)}
    merges = list(tokenizer.merges)  # ["tok1 tok2", ...]

    # Collect special/added tokens from the tokenizer's registered attrs.
    added_tokens = []
    if hasattr(tokenizer, "_special_token_attrs"):
        seen_ids = set()
        for attr in tokenizer._special_token_attrs:
            token = getattr(tokenizer, attr, None)
            if token is None:
                continue
            token_id = vocab.get(token)
            if token_id is not None and token_id not in seen_ids:
                added_tokens.append(
                    {
                        "id": token_id,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )
                seen_ids.add(token_id)
    added_tokens.sort(key=lambda x: x["id"])

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        # Llama3 uses byte-level BPE with tiktoken-style regex splitting.
        # The pre_tokenizer must be a Sequence: first split on the Llama 3
        # regex, then apply ByteLevel encoding (with use_regex=False so the
        # GPT-2 fallback regex is not applied a second time).
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": (
                            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                            r"|\p{N}{1,3}"
                            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                            r"|\s*[\r\n]+"
                            r"|\s+(?!\S)"
                            r"|\s+"
                        )
                    },
                    "behavior": "Removed",
                    "invert": True,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
