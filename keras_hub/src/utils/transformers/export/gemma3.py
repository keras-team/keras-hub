import keras.ops as ops


def get_gemma3_config(backbone):
    """Convert Keras Gemma3 config to Hugging Face config dictionary."""

    layer_types = []
    for i in range(backbone.num_layers):
        if backbone.use_sliding_window_attention and (i % 6 < 5):
            layer_types.append("sliding_attention")
        else:
            layer_types.append("full_attention")

    # Check if this is a vision model
    has_vision = backbone.vision_encoder is not None

    # Base text config
    text_config = {
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": backbone.head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "attention_dropout": backbone.dropout,
        "hidden_activation": "gelu_pytorch_tanh",
        "sliding_window": backbone.sliding_window_size,
        "_sliding_window_pattern": 6,
        "use_cache": True,
        "torch_dtype": backbone.dtype_policy.name,
        "layer_types": layer_types,
        "query_pre_attn_scalar": backbone.head_dim
        if backbone.query_head_dim_normalize
        else backbone.hidden_dim // backbone.num_query_heads,
    }

    if has_vision:
        # Vision + Text model
        vision_encoder = backbone.vision_encoder
        image_encoder = vision_encoder.get_layer("image_encoder")
        
        vision_config = {
            "image_size": image_encoder.image_size,
            "patch_size": image_encoder.patch_size,
            "num_attention_heads": image_encoder.num_heads,
            "hidden_size": image_encoder.hidden_dim,
            "num_hidden_layers": image_encoder.num_layers,
            "intermediate_size": image_encoder.intermediate_dim,
            "layer_norm_eps": image_encoder.layer_norm_epsilon,
            "model_type": "siglip_vision_model",
            "vision_use_head": False,
        }
        
        hf_config = {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": text_config,
            "vision_config": vision_config,
            "torch_dtype": backbone.dtype_policy.name,
        }
    else:
        # Text-only model
        hf_config = {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
        }
        hf_config.update(text_config)

    return hf_config


def get_gemma3_weights_map(backbone, include_lm_head=False):
    """Convert a Keras Gemma3 model to Hugging Face format.

    include_lm_head: If True, exports for CausalLM (with "model." prefix).
                    If False, exports for backbone only (without prefix).
    """

    def _convert_qkv_kernel(kernel, hidden_dim):
        """Helper to convert Q/K/V projection kernels to HF format.

        Args:
            kernel: The kernel weight tensor to convert.
            hidden_dim: The hidden dimension size for reshaping.

        Returns:
            Converted kernel in HF format.
        """
        kernel = ops.transpose(kernel, axes=(1, 0, 2))  # permute(1, 0, 2)
        kernel = ops.reshape(kernel, (hidden_dim, -1))
        kernel = ops.transpose(kernel)  # .T
        return kernel

    weights_dict = {}
    has_vision = backbone.vision_encoder is not None

    # For vision models: use "model.language_model." prefix
    # For text-only CausalLM: use "model." prefix
    # For backbone export: use no prefix
    if has_vision:
        prefix = "model.language_model."
    else:
        prefix = "model." if include_lm_head else ""

    # === Vision Encoder Weights (if present) ===
    if has_vision:
        vision_encoder = backbone.vision_encoder
        image_encoder = vision_encoder.get_layer("image_encoder")
        vision_output_encoder = vision_encoder.get_layer("vision_output_encoder")
        
        # Patch embedding
        patch_embedding = image_encoder.vision_embeddings.patch_embedding
        weights_dict["model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = ops.transpose(
            patch_embedding.weights[0], axes=(3, 2, 0, 1)
        )  # (H, W, C, out) -> (out, C, H, W)
        weights_dict["model.vision_tower.vision_model.embeddings.patch_embedding.bias"] = patch_embedding.weights[1]
        
        # Position embedding
        weights_dict["model.vision_tower.vision_model.embeddings.position_embedding.weight"] = (
            image_encoder.vision_embeddings.position_embedding.weights[0]
        )
        
        # Vision transformer layers
        for i in range(image_encoder.num_layers):
            resblock = image_encoder.resblocks[i]
            
            # Layer norms
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = (
                resblock.layer_norm_1.weights[0]  # gamma
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = (
                resblock.layer_norm_1.weights[1]  # beta
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = (
                resblock.layer_norm_2.weights[0]
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = (
                resblock.layer_norm_2.weights[1]
            )
            
            # Attention projections
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = (
                ops.transpose(resblock.attn.query_proj.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = (
                resblock.attn.query_proj.weights[1]
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = (
                ops.transpose(resblock.attn.key_proj.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = (
                resblock.attn.key_proj.weights[1]
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = (
                ops.transpose(resblock.attn.value_proj.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = (
                resblock.attn.value_proj.weights[1]
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                ops.transpose(resblock.attn.out_proj.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                resblock.attn.out_proj.weights[1]
            )
            
            # MLP layers
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (
                ops.transpose(resblock.mlp_dense_1.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (
                resblock.mlp_dense_1.weights[1]
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (
                ops.transpose(resblock.mlp_dense_2.weights[0])
            )
            weights_dict[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (
                resblock.mlp_dense_2.weights[1]
            )
        
        # Post-encoder layer norm
        weights_dict["model.vision_tower.vision_model.post_layernorm.weight"] = (
            image_encoder.encoder_layer_norm.weights[0]  # gamma
        )
        weights_dict["model.vision_tower.vision_model.post_layernorm.bias"] = (
            image_encoder.encoder_layer_norm.weights[1]  # beta
        )
        
        # Multi-modal projector
        weights_dict["model.multi_modal_projector.mm_soft_emb_norm.weight"] = (
            vision_output_encoder.vision_soft_embedding_norm.weights[0]  # scale
        )
        weights_dict["model.multi_modal_projector.mm_input_projection_weight"] = (
            vision_output_encoder.vision_input_projection.weights[0]  # kernel
        )

    # Token embeddings - use .weights[0] to get backend tensor
    token_embedding_layer = backbone.get_layer("token_embedding")
    token_embedding = token_embedding_layer.weights[0]
    weights_dict[f"{prefix}embed_tokens.weight"] = token_embedding

    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"decoder_block_{i}")

        # Attention query projection
        q_kernel = _convert_qkv_kernel(
            block.attention.query_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.q_proj.weight"] = q_kernel

        # Attention key projection
        k_kernel = _convert_qkv_kernel(
            block.attention.key_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.k_proj.weight"] = k_kernel

        # Attention value projection
        v_kernel = _convert_qkv_kernel(
            block.attention.value_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.v_proj.weight"] = v_kernel

        # Attention output projection
        o_kernel = block.attention.output_dense.weights[0]
        o_kernel = ops.transpose(o_kernel, axes=(2, 0, 1))  # permute(2, 0, 1)
        o_kernel = ops.reshape(o_kernel, (backbone.hidden_dim, -1))
        weights_dict[f"{prefix}layers.{i}.self_attn.o_proj.weight"] = o_kernel

        # Query and key normalization
        q_norm = block.attention.query_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.self_attn.q_norm.weight"] = q_norm

        k_norm = block.attention.key_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.self_attn.k_norm.weight"] = k_norm

        # MLP gate projection
        gate_kernel = block.gating_ffw.weights[0]
        gate_kernel = ops.transpose(gate_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.gate_proj.weight"] = gate_kernel

        # MLP up projection
        up_kernel = block.gating_ffw_2.weights[0]
        up_kernel = ops.transpose(up_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.up_proj.weight"] = up_kernel

        # MLP down projection
        down_kernel = block.ffw_linear.weights[0]
        down_kernel = ops.transpose(down_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.down_proj.weight"] = down_kernel

        # Pre-attention normalization
        input_layer_norm = block.pre_attention_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.input_layernorm.weight"] = (
            input_layer_norm
        )

        # Post-attention normalization
        if hasattr(block, "post_attention_norm"):
            post_attn_norm = block.post_attention_norm.weights[0]
            weights_dict[
                f"{prefix}layers.{i}.post_attention_layernorm.weight"
            ] = post_attn_norm
        # Pre-feedforward normalization
        pre_feedforward_layernorm = block.pre_ffw_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.pre_feedforward_layernorm.weight"] = (
            pre_feedforward_layernorm
        )
        # Post-feedforward normalization
        if hasattr(block, "post_ffw_norm"):
            post_feedforward_layernorm = block.post_ffw_norm.weights[0]
            weights_dict[
                f"{prefix}layers.{i}.post_feedforward_layernorm.weight"
            ] = post_feedforward_layernorm

    # Final normalization
    final_norm = backbone.get_layer("final_normalization").weights[0]
    weights_dict[f"{prefix}norm.weight"] = final_norm

    if include_lm_head and not token_embedding_layer.tie_weights:
        weights_dict["lm_head.weight"] = ops.transpose(
            token_embedding_layer.reverse_embeddings
        )

    return weights_dict


def get_gemma3_image_converter_config(backbone):
    """Generate preprocessor config for vision models.
    
    Returns None for text-only models.
    """
    if backbone.vision_encoder is None:
        return None
    
    vision_encoder = backbone.vision_encoder
    image_encoder = vision_encoder.get_layer("image_encoder")
    img_size = image_encoder.image_size
    
    preprocessor_config = {
        "image_processor_type": "Gemma3ImageProcessor",
        "do_resize": True,
        "size": {"height": img_size, "width": img_size},
        "do_rescale": True,
        "rescale_factor": 1 / 255,  # 0.00392156862745098
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        # Pan-and-scan disabled (single crop)
        "do_pan_and_scan": None,
        "pan_and_scan_min_crop_size": None,
        "pan_and_scan_max_num_crops": None,
        "pan_and_scan_min_ratio_to_activate": None,
    }
    return preprocessor_config


def get_gemma3_processor_config(backbone):
    """Generate processor config for vision models.
    
    Returns None for text-only models.
    """
    if backbone.vision_encoder is None:
        return None
    
    # Calculate image sequence length based on image size and patch size
    vision_encoder = backbone.vision_encoder
    image_encoder = vision_encoder.get_layer("image_encoder")
    img_size = image_encoder.image_size
    patch_size = image_encoder.patch_size
    image_seq_length = (img_size // patch_size) ** 2
    
    processor_config = {
        "processor_class": "Gemma3Processor",
        "image_seq_length": image_seq_length,
    }
    return processor_config


def get_gemma3_tokenizer_config(tokenizer):
    tokenizer_config = {
        "tokenizer_class": "GemmaTokenizer",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 1000000000000000019884624838656,
        "spaces_between_special_tokens": False,
        "use_default_system_prompt": False,
    }
    
    # Check if this is a vision-enabled tokenizer
    has_vision_tokens = (
        tokenizer.token_to_id("<start_of_image>") is not None
        and tokenizer.token_to_id("<end_of_image>") is not None
        and tokenizer.token_to_id("<image_soft_token>") is not None
    )
    
    # Add vision-specific fields if present
    if has_vision_tokens:
        tokenizer_config["processor_class"] = "Gemma3Processor"
        # Vision tokens as simple strings - HF auto-creates *_token_id attributes
        tokenizer_config["boi_token"] = "<start_of_image>"
        tokenizer_config["eoi_token"] = "<end_of_image>"
        tokenizer_config["image_token"] = "<image_soft_token>"
        # extra_special_tokens is required for HF to recognize these as special tokens
        tokenizer_config["extra_special_tokens"] = {
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "image_token": "<image_soft_token>",
        }
    
    # Add added_tokens_decoder
    added_tokens_decoder = {}
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<mask>",
        "[multimodal]",
        "<img>",
    ]
    
    # Add vision tokens if present
    if has_vision_tokens:
        special_tokens.extend(["<start_of_image>", "<end_of_image>", "<image_soft_token>"])
    
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            added_tokens_decoder[str(token_id)] = {
                "content": token,
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder
    return tokenizer_config
