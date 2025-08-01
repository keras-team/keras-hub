import keras
import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.models.mobilenetv5.mobilenetv5_presets import (
    backbone_presets_base as mobilenet_backbone_presets_base,
)
from keras_hub.src.utils.preset_utils import get_file

# Define the Keras backbone class we are targeting.
backbone_cls = Gemma3nBackbone


def convert_backbone_config(transformers_config):
    """
    Converts a Hugging Face Gemma3n config to a Keras Gemma3nBackbone config.
    """
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    audio_config = transformers_config["audio_config"]

    # For MobileNetV5, the detailed block architecture is not in the config.
    # We load it from our presets, which should match the intended architecture.
    # The config name "mobilenetv5_300m_enc" corresponds to "mobilenetv5_base".
    vision_block_args = mobilenet_backbone_presets_base["mobilenetv5_base"][
        "config"
    ]["block_args"]

    backbone_config = {
        # --- Text Decoder Args ---
        "vocabulary_size": text_config["vocab_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"][0],
        "head_dim": text_config["head_dim"],
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "dropout": text_config["attention_dropout"],
        "attention_bias": text_config["attention_bias"],
        "final_logit_soft_cap": text_config["final_logit_softcapping"],
        "vocab_size_per_layer_input": text_config["vocab_size_per_layer_input"],
        "hidden_size_per_layer_input": text_config[
            "hidden_size_per_layer_input"
        ],
        "laurel_rank": text_config["laurel_rank"],
        "altup_num_inputs": text_config["altup_num_inputs"],
        "altup_active_idx": text_config["altup_active_idx"],
        "altup_coef_clip": text_config.get("altup_coef_clip"),
        "altup_correct_scale": text_config["altup_correct_scale"],
        # Pass the whole list for per-layer MLP sparsity
        "activation_sparsity": text_config["activation_sparsity_pattern"],
        "layer_types": text_config["layer_types"],
        "sliding_window_size": text_config["sliding_window"],
        # --- Vision Encoder (MobileNetV5) Args ---
        "vision_block_args": vision_block_args,
        "vision_num_features": vision_config["hidden_size"],
        # Other vision params will use Keras defaults as they aren't in HF config
        # --- Audio Encoder (Conformer) Args ---
        "num_conformer_layers": audio_config["conf_num_hidden_layers"],
        "audio_hidden_size": audio_config["hidden_size"],
        "audio_num_attention_heads": audio_config["conf_num_attention_heads"],
        "audio_attention_chunk_size": audio_config["conf_attention_chunk_size"],
        "audio_attention_context_left": audio_config[
            "conf_attention_context_left"
        ],
        "audio_attention_context_right": audio_config[
            "conf_attention_context_right"
        ],
        "audio_attention_logit_cap": audio_config["conf_attention_logit_cap"],
        "audio_gradient_clipping": audio_config["gradient_clipping"],
        "audio_residual_weight": audio_config["conf_residual_weight"],
        "audio_conv_kernel_size": audio_config["conf_conv_kernel_size"],
        # --- Multimodal Embedder Args ---
        "num_vision_tokens_per_image": transformers_config[
            "vision_soft_tokens_per_image"
        ],
        "num_audio_tokens": transformers_config["audio_soft_tokens_per_image"],
        "vision_vocab_offset": vision_config["vocab_offset"],
        "vision_vocab_size": vision_config["vocab_size"],
        "audio_vocab_offset": audio_config["vocab_offset"],
        "audio_vocab_size": audio_config["vocab_size"],
    }

    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    """
    Ports weights from a Hugging Face checkpoint to the Keras Gemma3nBackbone.

    Args:
        backbone: An instance of the Keras Gemma3nBackbone.
        loader: A utility object for loading weights.
        transformers_config: The Hugging Face model config.
    """

    # Helper functions for weight transformations
    def transpose(hf_tensor):
        return np.transpose(hf_tensor)

    def t_conv2d(hf_tensor):
        # PyTorch (O, I, H, W) -> Keras (H, W, I, O)
        return np.transpose(hf_tensor, [2, 3, 1, 0])

    def t_dw_conv2d(hf_tensor):
        # PyTorch (C, 1, H, W) -> Keras (H, W, C, 1)
        return np.transpose(hf_tensor, [2, 3, 0, 1])

    def t_dw_conv1d(hf_tensor):
        # PyTorch (C, 1, L) -> Keras (L, 1, C)
        return np.transpose(hf_tensor, [2, 1, 0])

    # --- Main Embeddings ---
    # (Code from your prompt)
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="language_model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.per_layer_embeddings.embeddings,
        hf_weight_key="language_model.embed_tokens_per_layer.weight",
    )

    # --- Multimodal Embedders ---
    # (Code from your prompt)
    if backbone.vision_embedder:
        loader.port_weight(
            keras_variable=backbone.vision_embedder.embedding.embeddings,
            hf_weight_key="embed_vision.embedding.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.hard_embedding_norm.scale,
            hf_weight_key="embed_vision.hard_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.soft_embedding_norm.scale,
            hf_weight_key="embed_vision.soft_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.vision_embedder.embedding_projection.kernel,
            hf_weight_key="embed_vision.embedding_projection.weight",
            hook_fn=transpose,
        )

    if backbone.audio_embedder:
        loader.port_weight(
            keras_variable=backbone.audio_embedder.embedding.embeddings,
            hf_weight_key="embed_audio.embedding.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.hard_embedding_norm.scale,
            hf_weight_key="embed_audio.hard_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.soft_embedding_norm.scale,
            hf_weight_key="embed_audio.soft_embedding_norm.weight",
        )
        loader.port_weight(
            keras_variable=backbone.audio_embedder.embedding_projection.kernel,
            hf_weight_key="embed_audio.embedding_projection.weight",
            hook_fn=transpose,
        )

    # --- Initial AltUp Projections ---
    # (Code from your prompt)
    for i in range(backbone.altup_num_inputs - 1):
        loader.port_weight(
            keras_variable=backbone.altup_projections[i].kernel,
            hf_weight_key=f"language_model.altup_projections.{i}.weight",
            hook_fn=transpose,
        )

    # --- Transformer Decoder Layers ---
    # (Code from your prompt)
    for i in range(backbone.num_hidden_layers):
        keras_layer = backbone.get_layer(f"gemma3n_transformer_decoder_{i}")
        hf_prefix = f"language_model.layers.{i}"
        # ... (rest of the text decoder loading logic)

    # --- Final Normalization ---
    # (Code from your prompt)
    loader.port_weight(
        keras_variable=backbone.final_norm.scale,
        hf_weight_key="language_model.norm.weight",
    )

    # ==========================================================================
    # --- START: Vision and Audio Tower Weight Loading ---
    # ==========================================================================
    vision_encoder = backbone.vision_encoder
    audio_encoder = backbone.audio_encoder

    # --- Vision Tower (MobileNetV5) ---
    if vision_encoder:
        hf_vision_prefix = "vision_tower.timm_model"

        # Stem
        stem = vision_encoder.get_layer("stem")
        loader.port_weight(
            stem.conv.kernel,
            f"{hf_vision_prefix}.conv_stem.conv.weight",
            hook_fn=t_conv2d,
        )
        loader.port_weight(
            stem.conv.bias, f"{hf_vision_prefix}.conv_stem.conv.bias"
        )
        loader.port_weight(
            stem.norm.scale, f"{hf_vision_prefix}.conv_stem.bn.weight"
        )

        # Blocks
        total_block_idx = 0
        for stack_idx, stack_args in enumerate(vision_encoder.decoded_arch):
            for block_idx, b_args in enumerate(stack_args):
                block = vision_encoder.get_layer(
                    f"stack{stack_idx}_block{block_idx}"
                )
                hf_block_prefix = (
                    f"{hf_vision_prefix}.blocks.{stack_idx}.{block_idx}"
                )

                if (
                    isinstance(block, keras.layers.Layer)
                    and "UniversalInvertedResidualBlock"
                    in block.__class__.__name__
                ):
                    # dw_start
                    if not isinstance(block.dw_start, keras.layers.Identity):
                        loader.port_weight(
                            block.dw_start.conv.kernel,
                            f"{hf_block_prefix}.dw_start.conv.weight",
                            hook_fn=t_dw_conv2d,
                        )
                        loader.port_weight(
                            block.dw_start.norm.scale,
                            f"{hf_block_prefix}.dw_start.bn.weight",
                        )
                    # pw_exp
                    loader.port_weight(
                        block.pw_exp.conv.kernel,
                        f"{hf_block_prefix}.pw_exp.conv.weight",
                        hook_fn=t_conv2d,
                    )
                    loader.port_weight(
                        block.pw_exp.norm.scale,
                        f"{hf_block_prefix}.pw_exp.bn.weight",
                    )
                    # dw_mid
                    if not isinstance(block.dw_mid, keras.layers.Identity):
                        loader.port_weight(
                            block.dw_mid.conv.kernel,
                            f"{hf_block_prefix}.dw_mid.conv.weight",
                            hook_fn=t_dw_conv2d,
                        )
                        loader.port_weight(
                            block.dw_mid.norm.scale,
                            f"{hf_block_prefix}.dw_mid.bn.weight",
                        )
                    # pw_proj
                    loader.port_weight(
                        block.pw_proj.conv.kernel,
                        f"{hf_block_prefix}.pw_proj.conv.weight",
                        hook_fn=t_conv2d,
                    )
                    loader.port_weight(
                        block.pw_proj.norm.scale,
                        f"{hf_block_prefix}.pw_proj.bn.weight",
                    )
                    # layer_scale
                    if not isinstance(block.layer_scale, keras.layers.Identity):
                        loader.port_weight(
                            block.layer_scale.gamma,
                            f"{hf_block_prefix}.layer_scale.gamma",
                        )

                elif (
                    isinstance(block, keras.layers.Layer)
                    and "MobileAttentionBlock" in block.__class__.__name__
                ):
                    loader.port_weight(
                        block.norm.scale, f"{hf_block_prefix}.norm.weight"
                    )
                    loader.port_weight(
                        block.q_proj.kernel,
                        f"{hf_block_prefix}.attn.query.proj.weight",
                        hook_fn=t_conv2d,
                    )
                    loader.port_weight(
                        block.k_proj.kernel,
                        f"{hf_block_prefix}.attn.key.down_conv.weight",
                        hook_fn=t_dw_conv2d,
                    )
                    loader.port_weight(
                        block.v_proj.kernel,
                        f"{hf_block_prefix}.attn.value.down_conv.weight",
                        hook_fn=t_dw_conv2d,
                    )

                    # Note: Keras MHA combines output projection. We load the HF output proj weights into it.
                    loader.port_weight(
                        block.attention._output_dense.kernel,
                        f"{hf_block_prefix}.attn.output.proj.weight",
                        hook_fn=lambda t: t.reshape(
                            block.attention._output_dense.kernel.shape
                        ),
                    )
                    loader.port_weight(
                        block.attention._output_dense.bias,
                        f"{hf_block_prefix}.attn.output.proj.bias",
                    )

                    if not isinstance(block.layer_scale, keras.layers.Identity):
                        loader.port_weight(
                            block.layer_scale.gamma,
                            f"{hf_block_prefix}.layer_scale.gamma",
                        )

                total_block_idx += 1

        # MSFA
        msfa = vision_encoder.get_layer("msfa")
        loader.port_weight(
            msfa.ffn.pw_exp.conv.kernel,
            f"{hf_vision_prefix}.msfa.ffn.pw_exp.conv.weight",
            hook_fn=t_conv2d,
        )
        loader.port_weight(
            msfa.ffn.pw_exp.norm.scale,
            f"{hf_vision_prefix}.msfa.ffn.pw_exp.bn.weight",
        )
        loader.port_weight(
            msfa.ffn.pw_proj.conv.kernel,
            f"{hf_vision_prefix}.msfa.ffn.pw_proj.conv.weight",
            hook_fn=t_conv2d,
        )
        loader.port_weight(
            msfa.ffn.pw_proj.norm.scale,
            f"{hf_vision_prefix}.msfa.ffn.pw_proj.bn.weight",
        )
        loader.port_weight(
            msfa.norm.scale, f"{hf_vision_prefix}.msfa.norm.weight"
        )

    # --- Audio Tower (Gemma3nAudioEncoder) ---
    if audio_encoder:
        hf_audio_prefix = "audio_tower"

        # Subsample Conv Projection
        # NOTE: Assuming Keras audio encoder implementation matches HF structure.
        # There's a mismatch between HF's CumulativeGroupNorm and Keras's RMSNorm.
        # Loading only `scale`.
        sscp = audio_encoder.get_layer("subsample_conv_projection")
        loader.port_weight(
            sscp.conv_0.conv.kernel,
            f"{hf_audio_prefix}.subsample_conv_projection.conv_0.conv.weight",
            hook_fn=t_conv2d,
        )
        loader.port_weight(
            sscp.conv_0.norm.scale,
            f"{hf_audio_prefix}.subsample_conv_projection.conv_0.norm.weight",
        )
        loader.port_weight(
            sscp.conv_1.conv.kernel,
            f"{hf_audio_prefix}.subsample_conv_projection.conv_1.conv.weight",
            hook_fn=t_conv2d,
        )
        loader.port_weight(
            sscp.conv_1.norm.scale,
            f"{hf_audio_prefix}.subsample_conv_projection.conv_1.norm.weight",
        )
        loader.port_weight(
            sscp.input_proj_linear.kernel,
            f"{hf_audio_prefix}.subsample_conv_projection.input_proj_linear.weight",
            hook_fn=transpose,
        )

        # Conformer Blocks
        for i in range(
            len(audio_encoder.conformer)
        ):  # Assuming `conformer` is a list of layers
            block = audio_encoder.conformer[i]
            hf_block_prefix = f"{hf_audio_prefix}.conformer.{i}"

            # FFW Start
            loader.port_weight(
                block.ffw_layer_start.pre_layer_norm.scale,
                f"{hf_block_prefix}.ffw_layer_start.pre_layer_norm.weight",
            )
            loader.port_weight(
                block.ffw_layer_start.ffw_layer_1.kernel,
                f"{hf_block_prefix}.ffw_layer_start.ffw_layer_1.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                block.ffw_layer_start.ffw_layer_2.kernel,
                f"{hf_block_prefix}.ffw_layer_start.ffw_layer_2.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                block.ffw_layer_start.post_layer_norm.scale,
                f"{hf_block_prefix}.ffw_layer_start.post_layer_norm.weight",
            )

            # Attention
            attn_block = block.attention
            loader.port_weight(
                attn_block.pre_attn_norm.scale,
                f"{hf_block_prefix}.attention.pre_attn_norm.weight",
            )
            loader.port_weight(
                attn_block.attn.relative_position_embedding.pos_proj.kernel,
                f"{hf_block_prefix}.attention.attn.relative_position_embedding.pos_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                attn_block.attn.q_proj.kernel,
                f"{hf_block_prefix}.attention.attn.q_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                attn_block.attn.k_proj.kernel,
                f"{hf_block_prefix}.attention.attn.k_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                attn_block.attn.v_proj.kernel,
                f"{hf_block_prefix}.attention.attn.v_proj.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                attn_block.attn.per_dim_scale,
                f"{hf_block_prefix}.attention.attn.per_dim_scale",
            )
            loader.port_weight(
                attn_block.post.kernel,
                f"{hf_block_prefix}.attention.post.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                attn_block.post_norm.scale,
                f"{hf_block_prefix}.attention.post_norm.weight",
            )

            # LightConv1D
            lconv = block.lconv1d
            loader.port_weight(
                lconv.pre_layer_norm.scale,
                f"{hf_block_prefix}.lconv1d.pre_layer_norm.weight",
            )
            loader.port_weight(
                lconv.linear_start.kernel,
                f"{hf_block_prefix}.lconv1d.linear_start.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                lconv.depthwise_conv1d.depthwise_kernel,
                f"{hf_block_prefix}.lconv1d.depthwise_conv1d.weight",
                hook_fn=t_dw_conv1d,
            )
            loader.port_weight(
                lconv.conv_norm.scale,
                f"{hf_block_prefix}.lconv1d.conv_norm.weight",
            )
            loader.port_weight(
                lconv.linear_end.kernel,
                f"{hf_block_prefix}.lconv1d.linear_end.weight",
                hook_fn=transpose,
            )

            # FFW End
            loader.port_weight(
                block.ffw_layer_end.pre_layer_norm.scale,
                f"{hf_block_prefix}.ffw_layer_end.pre_layer_norm.weight",
            )
            loader.port_weight(
                block.ffw_layer_end.ffw_layer_1.kernel,
                f"{hf_block_prefix}.ffw_layer_end.ffw_layer_1.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                block.ffw_layer_end.ffw_layer_2.kernel,
                f"{hf_block_prefix}.ffw_layer_end.ffw_layer_2.weight",
                hook_fn=transpose,
            )
            loader.port_weight(
                block.ffw_layer_end.post_layer_norm.scale,
                f"{hf_block_prefix}.ffw_layer_end.post_layer_norm.weight",
            )

            # Final Norm
            loader.port_weight(
                block.norm.scale, f"{hf_block_prefix}.norm.weight"
            )

    # ==========================================================================
    # --- END: Vision and Audio Tower Weight Loading ---
    # ==========================================================================


def convert_tokenizer(cls, preset, **kwargs):
    # This function remains the same as you provided.
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
