import numpy as np

from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3OmniBackbone


def convert_backbone_config(transformers_config):
    """Convert HuggingFace Qwen3-Omni config to KerasHub config."""
    thinker_config = transformers_config.get("thinker_config", {})
    text_config = thinker_config.get("text_config", transformers_config)
    rope_scaling = text_config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
    backbone_config = {
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

    backbone_config["image_token_id"] = thinker_config.get(
        "image_token_id", 151655
    )
    backbone_config["video_token_id"] = thinker_config.get(
        "video_token_id", 151656
    )
    backbone_config["audio_token_id"] = thinker_config.get(
        "audio_token_id", 151675
    )

    audio_config = thinker_config.get("audio_config")
    if audio_config:
        from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
            Qwen3OmniAudioEncoder,
        )

        backbone_config["audio_encoder"] = Qwen3OmniAudioEncoder(
            num_mel_bins=audio_config["num_mel_bins"],
            d_model=audio_config["d_model"],
            encoder_layers=audio_config["encoder_layers"],
            encoder_attention_heads=audio_config["encoder_attention_heads"],
            encoder_ffn_dim=audio_config["encoder_ffn_dim"],
            output_dim=audio_config["output_dim"],
            downsample_hidden_size=audio_config.get(
                "downsample_hidden_size", 480
            ),
            max_source_positions=audio_config.get("max_source_positions", 1500),
            scale_embedding=audio_config.get("scale_embedding", False),
            activation_function=audio_config.get("activation_function", "gelu"),
            dropout=audio_config.get("dropout", 0.0),
        )

    vision_config = thinker_config.get("vision_config")
    if vision_config:
        from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
            Qwen3OmniVisionEncoder,
        )

        backbone_config["vision_encoder"] = Qwen3OmniVisionEncoder(
            depth=vision_config["depth"],
            hidden_size=vision_config["hidden_size"],
            num_heads=vision_config["num_heads"],
            patch_size=vision_config["patch_size"],
            out_hidden_size=vision_config.get(
                "out_hidden_size",
                text_config.get("hidden_size", 3584),
            ),
            hidden_act=vision_config.get("hidden_act", "gelu_pytorch_tanh"),
            intermediate_size=vision_config.get("intermediate_size", 4304),
            in_channels=vision_config.get("in_channels", 3),
            spatial_merge_size=vision_config.get("spatial_merge_size", 2),
            temporal_patch_size=vision_config.get("temporal_patch_size", 2),
        )

    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """Convert HF Thinker weights to KerasHub backbone."""

    # === Audio Encoder Weights ===
    if backbone.audio_encoder is not None:
        audio_enc = backbone.audio_encoder

        # Conv downsampling layers
        def conv2d_transpose(x, _):
            # PyTorch Conv2D: (out_channels, in_channels, H, W)
            # Keras Conv2D: (H, W, in_channels, out_channels)
            return np.transpose(x, (2, 3, 1, 0))

        loader.port_weight(
            keras_variable=audio_enc.conv2d1.kernel,
            hf_weight_key="audio_tower.conv2d1.weight",
            hook_fn=conv2d_transpose,
        )
        loader.port_weight(
            keras_variable=audio_enc.conv2d1.bias,
            hf_weight_key="audio_tower.conv2d1.bias",
        )
        loader.port_weight(
            keras_variable=audio_enc.conv2d2.kernel,
            hf_weight_key="audio_tower.conv2d2.weight",
            hook_fn=conv2d_transpose,
        )
        loader.port_weight(
            keras_variable=audio_enc.conv2d2.bias,
            hf_weight_key="audio_tower.conv2d2.bias",
        )
        loader.port_weight(
            keras_variable=audio_enc.conv2d3.kernel,
            hf_weight_key="audio_tower.conv2d3.weight",
            hook_fn=conv2d_transpose,
        )
        loader.port_weight(
            keras_variable=audio_enc.conv2d3.bias,
            hf_weight_key="audio_tower.conv2d3.bias",
        )

        loader.port_weight(
            keras_variable=audio_enc.conv_out.kernel,
            hf_weight_key="audio_tower.conv_out.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )

        # Transformer encoder layers
        def audio_transpose_and_reshape(x, shape):
            return np.reshape(np.transpose(x), shape)

        for i in range(audio_enc.encoder_layers_count):
            layer = audio_enc.encoder_layers[i]

            # Self-attention layer norm
            loader.port_weight(
                keras_variable=layer.self_attn_layer_norm.gamma,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn_layer_norm.weight",
            )
            loader.port_weight(
                keras_variable=layer.self_attn_layer_norm.beta,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn_layer_norm.bias",
            )

            # Attention QKV projections
            loader.port_weight(
                keras_variable=layer.self_attn._query_dense.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.q_proj.weight",
                hook_fn=audio_transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=layer.self_attn._query_dense.bias,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.q_proj.bias",
                hook_fn=lambda x, shape: np.reshape(x, shape),
            )
            loader.port_weight(
                keras_variable=layer.self_attn._key_dense.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.k_proj.weight",
                hook_fn=audio_transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=layer.self_attn._key_dense.bias,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.k_proj.bias",
                hook_fn=lambda x, shape: np.reshape(x, shape),
            )
            loader.port_weight(
                keras_variable=layer.self_attn._value_dense.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.v_proj.weight",
                hook_fn=audio_transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=layer.self_attn._value_dense.bias,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.v_proj.bias",
                hook_fn=lambda x, shape: np.reshape(x, shape),
            )
            loader.port_weight(
                keras_variable=layer.self_attn._output_dense.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.out_proj.weight",
                hook_fn=audio_transpose_and_reshape,
            )
            loader.port_weight(
                keras_variable=layer.self_attn._output_dense.bias,
                hf_weight_key=f"audio_tower.layers.{i}.self_attn.out_proj.bias",
            )

            # Feed-forward
            loader.port_weight(
                keras_variable=layer.final_layer_norm.gamma,
                hf_weight_key=f"audio_tower.layers.{i}.final_layer_norm.weight",
            )
            loader.port_weight(
                keras_variable=layer.final_layer_norm.beta,
                hf_weight_key=f"audio_tower.layers.{i}.final_layer_norm.bias",
            )
            loader.port_weight(
                keras_variable=layer.fc1.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.fc1.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=layer.fc1.bias,
                hf_weight_key=f"audio_tower.layers.{i}.fc1.bias",
            )
            loader.port_weight(
                keras_variable=layer.fc2.kernel,
                hf_weight_key=f"audio_tower.layers.{i}.fc2.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=layer.fc2.bias,
                hf_weight_key=f"audio_tower.layers.{i}.fc2.bias",
            )

        # Post-encoder layer norm
        loader.port_weight(
            keras_variable=audio_enc.ln_post.gamma,
            hf_weight_key="audio_tower.ln_post.weight",
        )
        loader.port_weight(
            keras_variable=audio_enc.ln_post.beta,
            hf_weight_key="audio_tower.ln_post.bias",
        )

        # Output projections
        loader.port_weight(
            keras_variable=audio_enc.proj1.kernel,
            hf_weight_key="audio_tower.proj1.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            keras_variable=audio_enc.proj1.bias,
            hf_weight_key="audio_tower.proj1.bias",
        )
        loader.port_weight(
            keras_variable=audio_enc.proj2.kernel,
            hf_weight_key="audio_tower.proj2.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            keras_variable=audio_enc.proj2.bias,
            hf_weight_key="audio_tower.proj2.bias",
        )

    # === Vision Encoder Weights ===
    if backbone.vision_encoder is not None:
        vision_enc = backbone.vision_encoder

        # Patch embedding (Conv3D)
        def conv3d_transpose(x, _):
            # PyTorch Conv3D: (out_channels, in_channels, D, H, W)
            # Keras Conv3D: (D, H, W, in_channels, out_channels)
            return np.transpose(x, (2, 3, 4, 1, 0))

        loader.port_weight(
            keras_variable=vision_enc.patch_embed.proj.kernel,
            hf_weight_key="visual.patch_embed.proj.weight",
            hook_fn=conv3d_transpose,
        )
        loader.port_weight(
            keras_variable=vision_enc.patch_embed.proj.bias,
            hf_weight_key="visual.patch_embed.proj.bias",
        )

        # Position embeddings
        loader.port_weight(
            keras_variable=vision_enc.pos_embed.embeddings,
            hf_weight_key="visual.pos_embed.weight",
        )

        # Vision transformer blocks
        for i in range(vision_enc.depth):
            block = vision_enc.blocks[i]

            # Layer norms
            loader.port_weight(
                keras_variable=block.norm1.gamma,
                hf_weight_key=f"visual.blocks.{i}.norm1.weight",
            )
            loader.port_weight(
                keras_variable=block.norm1.beta,
                hf_weight_key=f"visual.blocks.{i}.norm1.bias",
            )
            loader.port_weight(
                keras_variable=block.norm2.gamma,
                hf_weight_key=f"visual.blocks.{i}.norm2.weight",
            )
            loader.port_weight(
                keras_variable=block.norm2.beta,
                hf_weight_key=f"visual.blocks.{i}.norm2.bias",
            )

            # Attention QKV
            loader.port_weight(
                keras_variable=block.attn.qkv.kernel,
                hf_weight_key=f"visual.blocks.{i}.attn.qkv.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=block.attn.qkv.bias,
                hf_weight_key=f"visual.blocks.{i}.attn.qkv.bias",
            )
            loader.port_weight(
                keras_variable=block.attn.proj.kernel,
                hf_weight_key=f"visual.blocks.{i}.attn.proj.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=block.attn.proj.bias,
                hf_weight_key=f"visual.blocks.{i}.attn.proj.bias",
            )
            loader.port_weight(
                keras_variable=block.mlp.fc1.kernel,
                hf_weight_key=f"visual.blocks.{i}.mlp.linear_fc1.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=block.mlp.fc1.bias,
                hf_weight_key=f"visual.blocks.{i}.mlp.linear_fc1.bias",
            )
            loader.port_weight(
                keras_variable=block.mlp.fc2.kernel,
                hf_weight_key=f"visual.blocks.{i}.mlp.linear_fc2.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=block.mlp.fc2.bias,
                hf_weight_key=f"visual.blocks.{i}.mlp.linear_fc2.bias",
            )

        loader.port_weight(
            keras_variable=vision_enc.merger.ln_q.gamma,
            hf_weight_key="visual.merger.ln_q.weight",
        )
        loader.port_weight(
            keras_variable=vision_enc.merger.ln_q.beta,
            hf_weight_key="visual.merger.ln_q.bias",
        )
        loader.port_weight(
            keras_variable=vision_enc.merger.mlp_fc1.kernel,
            hf_weight_key="visual.merger.mlp.0.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            keras_variable=vision_enc.merger.mlp_fc1.bias,
            hf_weight_key="visual.merger.mlp.0.bias",
        )
        loader.port_weight(
            keras_variable=vision_enc.merger.mlp_fc2.kernel,
            hf_weight_key="visual.merger.mlp.2.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            keras_variable=vision_enc.merger.mlp_fc2.bias,
            hf_weight_key="visual.merger.mlp.2.bias",
        )

        # Deepstack mergers
        for j in range(len(vision_enc.merger_list)):
            merger = vision_enc.merger_list[j]
            loader.port_weight(
                keras_variable=merger.ln_q.gamma,
                hf_weight_key=f"visual.merger_list.{j}.ln_q.weight",
            )
            loader.port_weight(
                keras_variable=merger.ln_q.beta,
                hf_weight_key=f"visual.merger_list.{j}.ln_q.bias",
            )
            loader.port_weight(
                keras_variable=merger.mlp_fc1.kernel,
                hf_weight_key=f"visual.merger_list.{j}.mlp.0.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=merger.mlp_fc1.bias,
                hf_weight_key=f"visual.merger_list.{j}.mlp.0.bias",
            )
            loader.port_weight(
                keras_variable=merger.mlp_fc2.kernel,
                hf_weight_key=f"visual.merger_list.{j}.mlp.2.weight",
                hook_fn=lambda x, _: np.transpose(x, (1, 0)),
            )
            loader.port_weight(
                keras_variable=merger.mlp_fc2.bias,
                hf_weight_key=f"visual.merger_list.{j}.mlp.2.bias",
            )

    # === Text Transformer Weights ===
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
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
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense_layer_norm.scale,
            hf_weight_key=f"model.layers.{i}.self_attn.q_norm.weight",
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense_layer_norm.scale,
            hf_weight_key=f"model.layers.{i}.self_attn.k_norm.weight",
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
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
                hf_weight_key=f"model.layers.{i}.mlp.gate.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Batched experts: gate_up_proj and down_proj
            gate_up_proj_list = []
            down_proj_list = []
            for expert_idx in range(backbone.num_experts):
                gate_proj = loader.get_tensor(
                    f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight"
                )
                up_proj = loader.get_tensor(
                    f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight"
                )
                # Transpose to (hidden_dim, intermediate_dim)
                gate_proj = np.transpose(gate_proj, axes=(1, 0))
                up_proj = np.transpose(up_proj, axes=(1, 0))
                # Concatenate gate_proj and up_proj along the last dimension
                gate_up_proj = np.concatenate([gate_proj, up_proj], axis=-1)
                gate_up_proj_list.append(gate_up_proj)

                # Load down_proj for each expert
                down_proj = loader.get_tensor(
                    f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight"
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
                hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=decoder_layer._feedforward_output_dense.kernel,
                hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=decoder_layer._feedforward_gate_dense.kernel,
                hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
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


def load_image_converter_config(preset, transformers_config):
    """Load image converter config from a HuggingFace preset."""
    thinker_config = transformers_config.get("thinker_config", {})
    vision_config = thinker_config.get("vision_config")
    if vision_config:
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        image_mean = preprocessor_config.get(
            "image_mean", [0.48145466, 0.4578275, 0.40821073]
        )
        image_std = preprocessor_config.get(
            "image_std", [0.26862954, 0.26130258, 0.27577711]
        )
        rescale_factor = preprocessor_config.get("rescale_factor", 1 / 255)
        offset = [(-m / s) for m, s in zip(image_mean, image_std)]
        scale = [(s * rescale_factor) for s in image_std]
        image_size = vision_config.get("image_size", 448)
        return {
            "image_size": (image_size, image_size),
            "scale": scale,
            "offset": offset,
        }
    return None


def load_audio_converter_config(preset, transformers_config):
    """Load audio converter config from a HuggingFace preset."""
    thinker_config = transformers_config.get("thinker_config", {})
    audio_config = thinker_config.get("audio_config")
    if audio_config:
        return {
            "num_mels": audio_config.get("num_mel_bins", 128),
            "sampling_rate": 16000,
            "max_audio_length": 300,
        }
    return None


def convert_tokenizer(cls, preset, **kwargs):
    vocab = load_json(preset, "vocab.json")
    merges_file = get_file(preset, "merges.txt")
    with open(merges_file, "r") as f:
        merges = [line.strip() for line in f if line.strip()]
    tokenizer_config = load_json(preset, "tokenizer_config.json")
    special_tokens = []
    if "added_tokens_decoder" in tokenizer_config:
        for token_id, token_info in tokenizer_config[
            "added_tokens_decoder"
        ].items():
            content = token_info.get("content", "")
            if not content.startswith("<|reserved_special_token_"):
                special_tokens.append(content)
                if content not in vocab:
                    vocab[content] = int(token_id)
    kwargs.update({"unsplittable_tokens": special_tokens})
    return cls(vocabulary=vocab, merges=merges, **kwargs)
