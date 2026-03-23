import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3ASRBackbone


def convert_backbone_config(transformers_config):
    # The HF config nests audio and text configs under thinker_config.
    thinker_cfg = transformers_config.get("thinker_config", transformers_config)
    audio_cfg = thinker_cfg.get("audio_config", thinker_cfg)
    text_cfg = thinker_cfg.get("text_config", thinker_cfg)

    return {
        "vocabulary_size": text_cfg.get("vocab_size", 151936),
        "num_layers": text_cfg.get("num_hidden_layers", 28),
        "num_query_heads": text_cfg.get("num_attention_heads", 16),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 8),
        "head_dim": text_cfg.get("head_dim", 128),
        "hidden_dim": text_cfg.get("hidden_size", 2048),
        "intermediate_dim": text_cfg.get("intermediate_size", 6144),
        "num_mel_bins": audio_cfg.get("num_mel_bins", 128),
        "encoder_d_model": audio_cfg.get("d_model", 1024),
        "encoder_num_layers": audio_cfg.get("encoder_layers", 24),
        "encoder_attention_heads": audio_cfg.get("encoder_attention_heads", 16),
        "encoder_ffn_dim": audio_cfg.get("encoder_ffn_dim", 4096),
        "downsample_hidden_size": audio_cfg.get("downsample_hidden_size", 480),
        "n_window": audio_cfg.get("n_window", 50),
        "n_window_infer": audio_cfg.get("n_window_infer", 800),
        "max_source_positions": audio_cfg.get("max_source_positions", 1500),
        "audio_token_id": thinker_cfg.get("audio_token_id", 151676),
        "rope_max_wavelength": text_cfg.get("rope_theta", 1000000),
        "layer_norm_epsilon": text_cfg.get("rms_norm_eps", 1e-6),
        "tie_word_embeddings": thinker_cfg.get(
            "tie_word_embeddings",
            transformers_config.get("tie_word_embeddings", True),
        ),
        "sliding_window_size": text_cfg.get("sliding_window", 32768),
    }


def convert_weights(backbone, loader, transformers_config):
    # All weights are prefixed with "thinker." in the HF checkpoint.

    # --- Token embedding ---
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="thinker.model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="thinker.lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # --- Audio encoder: Conv2D downsampling layers ---
    audio_enc = backbone.get_layer("audio_encoder")

    def convert_conv2d(keras_layer, hf_prefix):
        """Convert HF Conv2D weights to Keras Conv2D format.

        HF shape: (out_channels, in_channels, kH, kW)
        Keras shape: (kH, kW, in_channels, out_channels)
        """
        loader.port_weight(
            keras_variable=keras_layer.kernel,
            hf_weight_key=f"{hf_prefix}.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, axes=(2, 3, 1, 0)
            ),
        )
        if keras_layer.use_bias:
            loader.port_weight(
                keras_variable=keras_layer.bias,
                hf_weight_key=f"{hf_prefix}.bias",
            )

    convert_conv2d(audio_enc.conv2d_1, "thinker.audio_tower.conv2d1")
    convert_conv2d(audio_enc.conv2d_2, "thinker.audio_tower.conv2d2")
    convert_conv2d(audio_enc.conv2d_3, "thinker.audio_tower.conv2d3")

    # Conv projection (Dense, no bias).
    loader.port_weight(
        keras_variable=audio_enc.conv_projection.kernel,
        hf_weight_key="thinker.audio_tower.conv_out.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    # --- Audio encoder: Transformer layers ---
    for i in range(audio_enc.num_encoder_layers):
        enc_layer = audio_enc._encoder_layers[i]
        hf_prefix = f"thinker.audio_tower.layers.{i}"

        # Pre-attention layer norm.
        loader.port_weight(
            keras_variable=enc_layer.self_attn_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.self_attn_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=enc_layer.self_attn_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.self_attn_layer_norm.bias",
        )

        # Self-attention projections.
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            keras_proj = getattr(enc_layer, proj_name)
            loader.port_weight(
                keras_variable=keras_proj.kernel,
                hf_weight_key=f"{hf_prefix}.self_attn.{proj_name}.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            if keras_proj.use_bias:
                loader.port_weight(
                    keras_variable=keras_proj.bias,
                    hf_weight_key=f"{hf_prefix}.self_attn.{proj_name}.bias",
                )

        # Post-attention (feedforward) layer norm.
        loader.port_weight(
            keras_variable=enc_layer.final_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}.final_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=enc_layer.final_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}.final_layer_norm.bias",
        )

        # Feedforward layers.
        loader.port_weight(
            keras_variable=enc_layer.fc1.kernel,
            hf_weight_key=f"{hf_prefix}.fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=enc_layer.fc1.bias,
            hf_weight_key=f"{hf_prefix}.fc1.bias",
        )
        loader.port_weight(
            keras_variable=enc_layer.fc2.kernel,
            hf_weight_key=f"{hf_prefix}.fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=enc_layer.fc2.bias,
            hf_weight_key=f"{hf_prefix}.fc2.bias",
        )

    # Post-encoder layer norm (ln_post in HF).
    loader.port_weight(
        keras_variable=audio_enc.layer_norm.gamma,
        hf_weight_key="thinker.audio_tower.ln_post.weight",
    )
    loader.port_weight(
        keras_variable=audio_enc.layer_norm.beta,
        hf_weight_key="thinker.audio_tower.ln_post.bias",
    )

    # Output projection (proj1 + GELU + proj2, inside audio_tower).
    if audio_enc.output_dim is not None:
        loader.port_weight(
            keras_variable=audio_enc.output_proj_1.kernel,
            hf_weight_key="thinker.audio_tower.proj1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=audio_enc.output_proj_1.bias,
            hf_weight_key="thinker.audio_tower.proj1.bias",
        )
        loader.port_weight(
            keras_variable=audio_enc.output_proj_2.kernel,
            hf_weight_key="thinker.audio_tower.proj2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=audio_enc.output_proj_2.bias,
            hf_weight_key="thinker.audio_tower.proj2.bias",
        )

    # --- Qwen3 decoder layers ---
    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm.
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"thinker.model.layers.{i}.input_layernorm.weight",
        )

        # Self-attention.
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense_layer_norm.scale,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.q_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense_layer_norm.scale,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.k_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
        )

        # MLP layers.
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"thinker.model.layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Post-attention layernorm.
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"thinker.model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization.
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="thinker.model.norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    merges = [" ".join(item) for item in merges]

    special_tokens = set()
    for token in tokenizer_config["added_tokens"]:
        if not token["content"].startswith("<|reserved_special_token_"):
            vocab[token["content"]] = token["id"]
            special_tokens.add(token["content"])

    kwargs.update(
        {
            "unsplittable_tokens": list(special_tokens),
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)
