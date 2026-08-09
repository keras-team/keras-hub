import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3ASRBackbone


def load_audio_converter_config(preset, transformers_config):
    audio_config = transformers_config.get("audio_config", {})
    return {
        "num_mels": audio_config.get("num_mel_bins", 128),
        "n_window": audio_config.get("n_window", 50),
    }


def convert_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    audio_config = transformers_config["audio_config"]

    rope_theta = text_config.get("rope_parameters", {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = text_config["rope_theta"]

    config = {
        "vocabulary_size": text_config["vocab_size"],
        "head_dim": text_config["head_dim"],
        "hidden_dim": text_config["hidden_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "intermediate_dim": text_config["intermediate_size"],
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "rope_max_wavelength": rope_theta,
        "sliding_window_size": text_config["sliding_window"]
        if text_config["use_sliding_window"]
        else None,
        "tie_word_embeddings": text_config["tie_word_embeddings"],
    }

    config.update(
        {
            "audio_num_mel_bins": audio_config["num_mel_bins"],
            "audio_num_layers": audio_config["encoder_layers"],
            "audio_num_attention_heads": audio_config[
                "encoder_attention_heads"
            ],
            "audio_intermediate_dim": audio_config["encoder_ffn_dim"],
            "audio_d_model": audio_config["d_model"],
            "audio_n_window": audio_config["n_window"],
            "audio_n_window_infer": audio_config["n_window_infer"],
            "audio_downsample_hidden_size": audio_config[
                "downsample_hidden_size"
            ],
            "audio_max_position_embeddings": audio_config[
                "max_position_embeddings"
            ],
            "audio_token_id": transformers_config["audio_token_id"],
        }
    )
    return config


def convert_weights(backbone, loader, transformers_config):
    def text_hf_key(suffix):
        return f"model.language_model.{suffix}"

    def audio_hf_key(suffix):
        return f"model.audio_tower.{suffix}"

    def proj_hf_key(suffix):
        return f"model.multi_modal_projector.{suffix}"

    # Text Backbone
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=text_hf_key("embed_tokens.weight"),
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

    def reshape_bias(x, shape):
        return np.reshape(x, shape)

    def transpose_output_kernel(x, shape):
        return np.transpose(
            np.reshape(x, (shape[2], shape[0], shape[1])),
            axes=(1, 2, 0),
        )

    def transpose_conv_out_kernel(x, shape):
        d_model = shape[1]
        channels = backbone.audio_encoder.downsample_hidden_size
        freq = shape[0] // channels
        x = np.reshape(x, (d_model, channels, freq))
        x = np.transpose(x, (0, 2, 1))
        x = np.reshape(x, (d_model, shape[0]))
        return np.transpose(x, (1, 0))

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=text_hf_key(f"layers.{i}.input_layernorm.weight"),
        )

        # Attention layers
        ## Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.q_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        ## Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.k_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        ## Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.v_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        ## Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.o_proj.weight"),
            hook_fn=transpose_and_reshape,
        )

        # QK Norm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense_layer_norm.scale,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.q_norm.weight"),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense_layer_norm.scale,
            hf_weight_key=text_hf_key(f"layers.{i}.self_attn.k_norm.weight"),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.mlp.up_proj.weight"),
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.mlp.down_proj.weight"),
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=text_hf_key(f"layers.{i}.mlp.gate_proj.weight"),
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=text_hf_key(
                f"layers.{i}.post_attention_layernorm.weight"
            ),
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key=text_hf_key("norm.weight"),
    )

    # Audio Encoder
    audio_encoder = backbone.audio_encoder

    # Stem
    loader.port_weight(
        keras_variable=audio_encoder.conv2d1.kernel,
        hf_weight_key=audio_hf_key("conv2d1.weight"),
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv2d1.bias,
        hf_weight_key=audio_hf_key("conv2d1.bias"),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv2d2.kernel,
        hf_weight_key=audio_hf_key("conv2d2.weight"),
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv2d2.bias,
        hf_weight_key=audio_hf_key("conv2d2.bias"),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv2d3.kernel,
        hf_weight_key=audio_hf_key("conv2d3.weight"),
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv2d3.bias,
        hf_weight_key=audio_hf_key("conv2d3.bias"),
    )
    loader.port_weight(
        keras_variable=audio_encoder.conv_out.kernel,
        hf_weight_key=audio_hf_key("conv_out.weight"),
        hook_fn=transpose_conv_out_kernel,
    )

    # Layers
    for i in range(audio_encoder.num_layers):
        block = audio_encoder.transformer_layers[i]

        # Self attention layernorm
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.gamma,
            hf_weight_key=audio_hf_key(
                f"layers.{i}.self_attn_layer_norm.weight"
            ),
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.beta,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn_layer_norm.bias"),
        )

        # Self attention
        ## Query
        loader.port_weight(
            keras_variable=block._self_attention_layer._query_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.q_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer._query_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.q_proj.bias"),
            hook_fn=reshape_bias,
        )
        ## Key
        loader.port_weight(
            keras_variable=block._self_attention_layer._key_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.k_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer._key_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.k_proj.bias"),
            hook_fn=reshape_bias,
        )
        ## Value
        loader.port_weight(
            keras_variable=block._self_attention_layer._value_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.v_proj.weight"),
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer._value_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.v_proj.bias"),
            hook_fn=reshape_bias,
        )
        ## Output
        loader.port_weight(
            keras_variable=block._self_attention_layer._output_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.out_proj.weight"),
            hook_fn=transpose_output_kernel,
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer._output_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.self_attn.out_proj.bias"),
        )

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.gamma,
            hf_weight_key=audio_hf_key(f"layers.{i}.final_layer_norm.weight"),
        )
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.beta,
            hf_weight_key=audio_hf_key(f"layers.{i}.final_layer_norm.bias"),
        )

        # Feedforward
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.fc1.weight"),
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.fc1.bias"),
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.kernel,
            hf_weight_key=audio_hf_key(f"layers.{i}.fc2.weight"),
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.bias,
            hf_weight_key=audio_hf_key(f"layers.{i}.fc2.bias"),
        )

    # Post LN
    loader.port_weight(
        keras_variable=audio_encoder.ln_post.gamma,
        hf_weight_key=audio_hf_key("ln_post.weight"),
    )
    loader.port_weight(
        keras_variable=audio_encoder.ln_post.beta,
        hf_weight_key=audio_hf_key("ln_post.bias"),
    )

    # Projector
    projector = backbone.projector
    loader.port_weight(
        keras_variable=projector.linear_1.kernel,
        hf_weight_key=proj_hf_key("linear_1.weight"),
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=projector.linear_1.bias,
        hf_weight_key=proj_hf_key("linear_1.bias"),
    )
    loader.port_weight(
        keras_variable=projector.linear_2.kernel,
        hf_weight_key=proj_hf_key("linear_2.weight"),
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=projector.linear_2.bias,
        hf_weight_key=proj_hf_key("linear_2.bias"),
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
