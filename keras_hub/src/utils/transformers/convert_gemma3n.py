import types

import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
    convert_arch_def_to_stackwise,
)
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Gemma3nBackbone


MOBILENETV5_300M_ENC_ARCH_DEF = [
    # Stage 0: 128x128 in
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    # Stage 1: 256x256 in
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
    ],
    # Stage 2: 640x640 in
    [
        "uir_r1_a5_k5_s2_e6_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a0_k0_s1_e1_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
    ],
    # Stage 3: 1280x1280 in
    [
        "uir_r1_a5_k5_s2_e6_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
    ],
]

mobilenetv5_config = convert_arch_def_to_stackwise(
    MOBILENETV5_300M_ENC_ARCH_DEF
)
mobilenetv5_config.update(
    {
        "stem_size": 64,
        "num_features": 2048,
        "norm_layer": "rms_norm",
        "act_layer": "gelu",
        "use_msfa": True,
        "layer_scale_init_value": 1e-5,
    }
)
MODEL_CONFIGS = {"mobilenetv5_300m_enc": mobilenetv5_config}


def convert_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    transformers_audio_config = transformers_config["audio_config"]
    audio_config = {
        "hidden_size": transformers_audio_config["hidden_size"],
        "input_feat_size": transformers_audio_config["input_feat_size"],
        "sscp_conv_channel_size": transformers_audio_config[
            "sscp_conv_channel_size"
        ],
        "sscp_conv_kernel_size": transformers_audio_config[
            "sscp_conv_kernel_size"
        ],
        "sscp_conv_stride_size": transformers_audio_config[
            "sscp_conv_stride_size"
        ],
        "sscp_conv_group_norm_eps": transformers_audio_config[
            "sscp_conv_group_norm_eps"
        ],
        "num_hidden_layers": transformers_audio_config[
            "conf_num_hidden_layers"
        ],
        "rms_norm_eps": transformers_audio_config["rms_norm_eps"],
        "gradient_clipping": transformers_audio_config["gradient_clipping"],
        "residual_weight": transformers_audio_config["conf_residual_weight"],
        "num_attention_heads": transformers_audio_config[
            "conf_num_attention_heads"
        ],
        "attention_chunk_size": transformers_audio_config[
            "conf_attention_chunk_size"
        ],
        "num_attention_context_right": transformers_audio_config[
            "conf_attention_context_right"
        ],
        "num_attention_context_left": transformers_audio_config[
            "conf_attention_context_left"
        ],
        "attention_logit_cap": transformers_audio_config[
            "conf_attention_logit_cap"
        ],
        "conv_kernel_size": transformers_audio_config["conf_conv_kernel_size"],
        "reduction_factor": transformers_audio_config["conf_reduction_factor"],
    }

    vision_encoder_config = MODEL_CONFIGS["mobilenetv5_300m_enc"].copy()
    vision_encoder_config["image_shape"] = (768, 768, 3)

    if text_config.get("hidden_activation") == "gelu_pytorch_tanh":
        hidden_activation = "gelu_approximate"
    else:
        hidden_activation = text_config.get(
            "hidden_activation", "gelu_approximate"
        )

    return {
        "text_vocab_size": text_config["vocab_size"],
        "text_hidden_size": text_config["hidden_size"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "pad_token_id": 0,
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "head_dim": text_config["head_dim"],
        "intermediate_size": text_config["intermediate_size"],
        "hidden_activation": hidden_activation,
        "layer_types": text_config["layer_types"],
        "sliding_window": text_config["sliding_window"],
        "rope_theta": text_config["rope_theta"],
        "max_position_embeddings": text_config["max_position_embeddings"],
        "vocab_size_per_layer_input": text_config["vocab_size_per_layer_input"],
        "hidden_size_per_layer_input": text_config[
            "hidden_size_per_layer_input"
        ],
        "altup_num_inputs": text_config["altup_num_inputs"],
        "laurel_rank": text_config["laurel_rank"],
        "attention_bias": text_config["attention_bias"],
        "attention_dropout": text_config["attention_dropout"],
        "rope_scaling": text_config.get("rope_scaling"),
        "activation_sparsity_pattern": text_config.get(
            "activation_sparsity_pattern"
        ),
        "altup_coef_clip": text_config["altup_coef_clip"],
        "altup_active_idx": text_config["altup_active_idx"],
        "altup_correct_scale": text_config["altup_correct_scale"],
        "num_kv_shared_layers": text_config["num_kv_shared_layers"],
        "final_logit_soft_cap": text_config.get("final_logit_softcapping"),
        "vision_encoder_config": vision_encoder_config,
        "vision_hidden_size": vision_config["hidden_size"],
        "vision_vocab_size": vision_config["vocab_size"],
        "vision_vocab_offset": vision_config["vocab_offset"],
        "vision_soft_tokens_per_image": transformers_config[
            "vision_soft_tokens_per_image"
        ],
        "image_token_id": transformers_config["image_token_id"],
        "audio_encoder_config": audio_config,
        "audio_hidden_size": transformers_audio_config["hidden_size"],
        "audio_vocab_size": transformers_audio_config["vocab_size"],
        "audio_vocab_offset": transformers_audio_config["vocab_offset"],
        "audio_soft_tokens_per_image": transformers_config[
            "audio_soft_tokens_per_image"
        ],
        "audio_token_id": transformers_config["audio_token_id"],
        "rms_norm_eps": text_config["rms_norm_eps"],
    }


def _port_rms_norm(loader, layer, hf_prefix):
    key = f"{hf_prefix}.weight"
    # RmsNorm2d uses 'gamma', Gemma3nRMSNorm uses 'scale'
    if hasattr(layer, "gamma"):
        weight_attr = layer.gamma
    elif hasattr(layer, "scale"):
        weight_attr = layer.scale
    else:
        raise AttributeError(
            f"Layer {layer.name} has neither 'gamma' nor 'scale' attribute"
        )
    loader.port_weight(weight_attr, key)


def _port_bn(loader, layer, hf_prefix):
    loader.port_weight(layer.gamma, f"{hf_prefix}.weight")
    loader.port_weight(layer.beta, f"{hf_prefix}.bias")
    loader.port_weight(layer.moving_mean, f"{hf_prefix}.running_mean")
    loader.port_weight(layer.moving_variance, f"{hf_prefix}.running_var")


def _port_cna(loader, cna_layer, hf_conv_prefix, hf_norm_prefix):
    import keras

    if isinstance(cna_layer.conv, keras.layers.DepthwiseConv2D):
        loader.port_weight(
            cna_layer.conv.kernel,
            f"{hf_conv_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 0, 1)),
        )
    else:
        loader.port_weight(
            cna_layer.conv.kernel,
            f"{hf_conv_prefix}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

    try:
        loader.get_tensor(f"{hf_norm_prefix}.running_mean")
        _port_bn(loader, cna_layer.norm, hf_norm_prefix)
    except (KeyError, ValueError):
        _port_rms_norm(loader, cna_layer.norm, hf_norm_prefix)


def _port_attn(loader, attn_layer, hf_attn_prefix):
    loader.port_weight(
        attn_layer.query_layers[-1].kernel,
        f"{hf_attn_prefix}.query.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    if len(attn_layer.key_layers) > 1:
        loader.port_weight(
            attn_layer.key_layers[0].kernel,
            f"{hf_attn_prefix}.key.down_conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 0, 1)),
        )
        key_norm_layer = attn_layer.key_layers[1]
        try:
            loader.get_tensor(f"{hf_attn_prefix}.key.norm.running_mean")
            _port_bn(loader, key_norm_layer, f"{hf_attn_prefix}.key.norm")
        except (KeyError, ValueError):
            _port_rms_norm(loader, key_norm_layer, f"{hf_attn_prefix}.key.norm")

    loader.port_weight(
        attn_layer.key_layers[-1].kernel,
        f"{hf_attn_prefix}.key.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    if len(attn_layer.value_layers) > 1:
        loader.port_weight(
            attn_layer.value_layers[0].kernel,
            f"{hf_attn_prefix}.value.down_conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 0, 1)),
        )
        value_norm_layer = attn_layer.value_layers[1]
        try:
            loader.get_tensor(f"{hf_attn_prefix}.value.norm.running_mean")
            _port_bn(loader, value_norm_layer, f"{hf_attn_prefix}.value.norm")
        except (KeyError, ValueError):
            _port_rms_norm(
                loader, value_norm_layer, f"{hf_attn_prefix}.value.norm"
            )

    loader.port_weight(
        attn_layer.value_layers[-1].kernel,
        f"{hf_attn_prefix}.value.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    loader.port_weight(
        attn_layer.output_proj_layers[-2].kernel,
        f"{hf_attn_prefix}.output.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )


def _port_vision_tower(loader, keras_model):
    from keras_hub.src.models.mobilenetv5.mobilenetv5_attention import (
        MobileAttention,
    )
    from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import EdgeResidual
    from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import (
        UniversalInvertedResidual,
    )

    backbone = keras_model.vision_encoder
    hf_prefix = "model.vision_tower.timm_model"

    stem_layer = backbone.get_layer("conv_stem")
    _port_cna(
        loader,
        stem_layer,
        f"{hf_prefix}.conv_stem.conv",
        f"{hf_prefix}.conv_stem.bn",
    )

    block_layers = [
        layer
        for layer in backbone.layers
        if isinstance(
            layer,
            (EdgeResidual, UniversalInvertedResidual, MobileAttention),
        )
    ]
    block_counter = 0
    for stack_idx in range(len(backbone.stackwise_num_blocks)):
        for block_idx_in_stage in range(
            backbone.stackwise_num_blocks[stack_idx]
        ):
            block = block_layers[block_counter]
            block_prefix = (
                f"{hf_prefix}.blocks.{stack_idx}.{block_idx_in_stage}"
            )

            if isinstance(block, EdgeResidual):
                _port_cna(
                    loader,
                    block.conv_exp,
                    f"{block_prefix}.conv_exp",
                    f"{block_prefix}.bn1",
                )
                _port_cna(
                    loader,
                    block.conv_pwl,
                    f"{block_prefix}.conv_pwl",
                    f"{block_prefix}.bn2",
                )
            elif isinstance(block, UniversalInvertedResidual):
                if hasattr(block, "dw_start") and not isinstance(
                    block.dw_start, types.FunctionType
                ):
                    _port_cna(
                        loader,
                        block.dw_start,
                        f"{block_prefix}.dw_start.conv",
                        f"{block_prefix}.dw_start.bn",
                    )
                _port_cna(
                    loader,
                    block.pw_exp,
                    f"{block_prefix}.pw_exp.conv",
                    f"{block_prefix}.pw_exp.bn",
                )
                if hasattr(block, "dw_mid") and not isinstance(
                    block.dw_mid, types.FunctionType
                ):
                    _port_cna(
                        loader,
                        block.dw_mid,
                        f"{block_prefix}.dw_mid.conv",
                        f"{block_prefix}.dw_mid.bn",
                    )
                _port_cna(
                    loader,
                    block.pw_proj,
                    f"{block_prefix}.pw_proj.conv",
                    f"{block_prefix}.pw_proj.bn",
                )
                gamma_key = f"{block_prefix}.layer_scale.gamma"
                try:
                    loader.port_weight(block.layer_scale.gamma, gamma_key)
                except (KeyError, ValueError):
                    pass
            elif isinstance(block, MobileAttention):
                _port_rms_norm(loader, block.norm, f"{block_prefix}.norm")
                gamma_key = f"{block_prefix}.layer_scale.gamma"
                try:
                    loader.port_weight(block.layer_scale.gamma, gamma_key)
                except (KeyError, ValueError):
                    pass
                attn_prefix = f"{block_prefix}.attn"
                _port_attn(loader, block.attn, attn_prefix)
            block_counter += 1

    try:
        msfa_layer = backbone.get_layer("msfa")
        msfa_prefix = f"{hf_prefix}.msfa"
        ffn = msfa_layer.ffn
        _port_cna(
            loader,
            ffn.pw_exp,
            f"{msfa_prefix}.ffn.pw_exp.conv",
            f"{msfa_prefix}.ffn.pw_exp.bn",
        )
        _port_cna(
            loader,
            ffn.pw_proj,
            f"{msfa_prefix}.ffn.pw_proj.conv",
            f"{msfa_prefix}.ffn.pw_proj.bn",
        )
        _port_rms_norm(loader, msfa_layer.norm, f"{msfa_prefix}.norm")
    except ValueError:
        pass


def _port_language_model(loader, keras_model):
    lm = keras_model.language_model
    hf_prefix = "model.language_model"

    loader.port_weight(
        lm.embed_tokens.embedding.embeddings,
        f"{hf_prefix}.embed_tokens.weight",
    )
    _port_rms_norm(loader, lm.final_normalization, f"{hf_prefix}.norm")
    loader.port_weight(
        lm.embed_tokens_per_layer.embedding.embeddings,
        f"{hf_prefix}.embed_tokens_per_layer.weight",
    )
    loader.port_weight(
        lm.per_layer_model_projection.kernel,
        f"{hf_prefix}.per_layer_model_projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )
    _port_rms_norm(
        loader,
        lm.per_layer_projection_norm,
        f"{hf_prefix}.per_layer_projection_norm",
    )

    for i in range(len(lm.altup_projections)):
        loader.port_weight(
            lm.altup_projections[i].kernel,
            f"{hf_prefix}.altup_projections.{i}.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
    for i in range(len(lm.altup_unembed_projections)):
        loader.port_weight(
            lm.altup_unembed_projections[i].kernel,
            f"{hf_prefix}.altup_unembed_projections.{i}.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )

    for i in range(len(lm.transformer_layers)):
        layer = lm.transformer_layers[i]
        layer_prefix = f"{hf_prefix}.layers.{i}"

        loader.port_weight(
            layer.attention.q_proj.kernel,
            f"{layer_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.attention.k_proj.kernel,
            f"{layer_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.attention.v_proj.kernel,
            f"{layer_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.attention.o_proj.kernel,
            f"{layer_prefix}.self_attn.o_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader, layer.attention.q_norm, f"{layer_prefix}.self_attn.q_norm"
        )
        _port_rms_norm(
            loader, layer.attention.k_norm, f"{layer_prefix}.self_attn.k_norm"
        )

        loader.port_weight(
            layer.mlp.gate_proj.kernel,
            f"{layer_prefix}.mlp.gate_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.mlp.up_proj.kernel,
            f"{layer_prefix}.mlp.up_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.mlp.down_proj.kernel,
            f"{layer_prefix}.mlp.down_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )

        _port_rms_norm(
            loader, layer.input_layernorm, f"{layer_prefix}.input_layernorm"
        )
        _port_rms_norm(
            loader,
            layer.post_attention_layernorm,
            f"{layer_prefix}.post_attention_layernorm",
        )
        _port_rms_norm(
            loader,
            layer.pre_feedforward_layernorm,
            f"{layer_prefix}.pre_feedforward_layernorm",
        )
        _port_rms_norm(
            loader,
            layer.post_feedforward_layernorm,
            f"{layer_prefix}.post_feedforward_layernorm",
        )

        altup_prefix = f"{layer_prefix}.altup"
        loader.port_weight(
            layer.altup.correction_coefs.kernel,
            f"{altup_prefix}.correction_coefs.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.altup.prediction_coefs.kernel,
            f"{altup_prefix}.prediction_coefs.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.altup.modality_router.kernel,
            f"{altup_prefix}.modality_router.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader, layer.altup.router_norm, f"{altup_prefix}.router_norm"
        )

        if layer.altup.altup_correct_scale:
            loader.port_weight(
                layer.altup.correct_output_scale,
                f"{altup_prefix}.correct_output_scale",
            )

        laurel_prefix = f"{layer_prefix}.laurel"
        loader.port_weight(
            layer.laurel.linear_left.kernel,
            f"{laurel_prefix}.linear_left.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.laurel.linear_right.kernel,
            f"{laurel_prefix}.linear_right.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader,
            layer.laurel.post_laurel_norm,
            f"{laurel_prefix}.post_laurel_norm",
        )

        loader.port_weight(
            layer.per_layer_input_gate.kernel,
            f"{layer_prefix}.per_layer_input_gate.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            layer.per_layer_projection.kernel,
            f"{layer_prefix}.per_layer_projection.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader,
            layer.post_per_layer_input_norm,
            f"{layer_prefix}.post_per_layer_input_norm",
        )


def _port_audio_tower(loader, keras_model):
    audio_encoder = keras_model.audio_encoder
    hf_prefix = "model.audio_tower"

    ssp = audio_encoder.subsample_conv_projection
    ssp_prefix = f"{hf_prefix}.subsample_conv_projection"

    loader.port_weight(
        ssp.conv_0.conv.kernel,
        f"{ssp_prefix}.conv_0.conv.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        ssp.conv_0.norm.scale,
        f"{ssp_prefix}.conv_0.norm.weight",
    )
    loader.port_weight(
        ssp.conv_1.conv.kernel,
        f"{ssp_prefix}.conv_1.conv.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        ssp.conv_1.norm.scale,
        f"{ssp_prefix}.conv_1.norm.weight",
    )
    loader.port_weight(
        ssp.input_proj_linear.kernel,
        f"{ssp_prefix}.input_proj_linear.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )

    for i in range(len(audio_encoder.conformer)):
        block = audio_encoder.conformer[i]
        block_prefix = f"{hf_prefix}.conformer.{i}"

        ffw_start_prefix = f"{block_prefix}.ffw_layer_start"
        _port_rms_norm(
            loader,
            block.ffw_layer_start.pre_layer_norm,
            f"{ffw_start_prefix}.pre_layer_norm",
        )
        loader.port_weight(
            block.ffw_layer_start.ffw_layer_1.kernel,
            f"{ffw_start_prefix}.ffw_layer_1.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.ffw_layer_start.ffw_layer_2.kernel,
            f"{ffw_start_prefix}.ffw_layer_2.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader,
            block.ffw_layer_start.post_layer_norm,
            f"{ffw_start_prefix}.post_layer_norm",
        )

        attn_prefix = f"{block_prefix}.attention"
        _port_rms_norm(
            loader,
            block.attention.pre_attn_norm,
            f"{attn_prefix}.pre_attn_norm",
        )
        loader.port_weight(
            block.attention.attn.per_dim_scale,
            f"{attn_prefix}.attn.per_dim_scale",
        )
        loader.port_weight(
            block.attention.attn.relative_position_embedding.pos_proj.kernel,
            f"{attn_prefix}.attn.relative_position_embedding.pos_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.attention.attn.q_proj.kernel,
            f"{attn_prefix}.attn.q_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.attention.attn.k_proj.kernel,
            f"{attn_prefix}.attn.k_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.attention.attn.v_proj.kernel,
            f"{attn_prefix}.attn.v_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.attention.post.kernel,
            f"{attn_prefix}.post.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader, block.attention.post_norm, f"{attn_prefix}.post_norm"
        )

        lconv_prefix = f"{block_prefix}.lconv1d"
        _port_rms_norm(
            loader,
            block.lconv1d.pre_layer_norm,
            f"{lconv_prefix}.pre_layer_norm",
        )
        loader.port_weight(
            block.lconv1d.linear_start.kernel,
            f"{lconv_prefix}.linear_start.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.lconv1d.depthwise_conv1d.kernel,
            f"{lconv_prefix}.depthwise_conv1d.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 0, 1)),
        )
        _port_rms_norm(
            loader, block.lconv1d.conv_norm, f"{lconv_prefix}.conv_norm"
        )
        loader.port_weight(
            block.lconv1d.linear_end.kernel,
            f"{lconv_prefix}.linear_end.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )

        ffw_end_prefix = f"{block_prefix}.ffw_layer_end"
        _port_rms_norm(
            loader,
            block.ffw_layer_end.pre_layer_norm,
            f"{ffw_end_prefix}.pre_layer_norm",
        )
        loader.port_weight(
            block.ffw_layer_end.ffw_layer_1.kernel,
            f"{ffw_end_prefix}.ffw_layer_1.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        loader.port_weight(
            block.ffw_layer_end.ffw_layer_2.kernel,
            f"{ffw_end_prefix}.ffw_layer_2.weight",
            hook_fn=lambda x, _: np.transpose(x, (1, 0)),
        )
        _port_rms_norm(
            loader,
            block.ffw_layer_end.post_layer_norm,
            f"{ffw_end_prefix}.post_layer_norm",
        )
        _port_rms_norm(loader, block.norm, f"{block_prefix}.norm")


def _port_multimodal_embedders(loader, keras_model):
    vision_prefix = "model.embed_vision"
    loader.port_weight(
        keras_model.embed_vision.embedding.embeddings,
        f"{vision_prefix}.embedding.weight",
    )
    _port_rms_norm(
        loader,
        keras_model.embed_vision.hard_embedding_norm,
        f"{vision_prefix}.hard_embedding_norm",
    )
    _port_rms_norm(
        loader,
        keras_model.embed_vision.soft_embedding_norm,
        f"{vision_prefix}.soft_embedding_norm",
    )
    loader.port_weight(
        keras_model.embed_vision.embedding_projection.kernel,
        f"{vision_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )

    audio_prefix = "model.embed_audio"
    loader.port_weight(
        keras_model.embed_audio.embedding.embeddings,
        f"{audio_prefix}.embedding.weight",
    )
    _port_rms_norm(
        loader,
        keras_model.embed_audio.hard_embedding_norm,
        f"{audio_prefix}.hard_embedding_norm",
    )
    _port_rms_norm(
        loader,
        keras_model.embed_audio.soft_embedding_norm,
        f"{audio_prefix}.soft_embedding_norm",
    )
    loader.port_weight(
        keras_model.embed_audio.embedding_projection.kernel,
        f"{audio_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )


def convert_weights(backbone, loader, transformers_config):
    _port_vision_tower(loader, backbone)
    _port_language_model(loader, backbone)
    _port_audio_tower(loader, backbone)
    _port_multimodal_embedders(loader, backbone)
    return backbone


def load_image_converter_config(preset, transformers_config):
    try:
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        image_mean = preprocessor_config.get("image_mean", [0.5, 0.5, 0.5])
        image_std = preprocessor_config.get("image_std", [0.5, 0.5, 0.5])
        do_rescale = preprocessor_config.get("do_rescale", True)
        do_normalize = preprocessor_config.get("do_normalize", False)
        rescale_factor = preprocessor_config.get("rescale_factor", 1.0 / 255.0)
        image_size = preprocessor_config.get(
            "size", {"height": 768, "width": 768}
        )
        scale = rescale_factor if do_rescale else None
        offset = None
        if do_normalize:
            # Match HF behavior:
            # x = (x * rescale_factor - mean) / std when normalize=True
            if do_rescale:
                scale = [rescale_factor / s for s in image_std]
            else:
                scale = [1.0 / s for s in image_std]
            offset = [(-m / s) for m, s in zip(image_mean, image_std)]
        return {
            "image_size": (
                image_size.get("height", 768),
                image_size.get("width", 768),
            ),
            "scale": scale,
            "offset": offset,
        }
    except Exception:
        return None


def load_audio_converter_config(preset, transformers_config):
    try:
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        feature_extractor = preprocessor_config.get("feature_extractor", {})

        return {
            "feature_size": feature_extractor.get("feature_size", 128),
            "sampling_rate": feature_extractor.get("sampling_rate", 16000),
            "padding_value": feature_extractor.get("padding_value", 0.0),
            "return_attention_mask": feature_extractor.get(
                "return_attention_mask", True
            ),
            "frame_length_ms": feature_extractor.get("frame_length_ms", 32.0),
            "hop_length_ms": feature_extractor.get("hop_length_ms", 10.0),
            "min_frequency": feature_extractor.get("min_frequency", 125.0),
            "max_frequency": feature_extractor.get("max_frequency", 7600.0),
            "preemphasis": feature_extractor.get("preemphasis", 0.97),
            "preemphasis_htk_flavor": feature_extractor.get(
                "preemphasis_htk_flavor", True
            ),
            "fft_overdrive": feature_extractor.get("fft_overdrive", True),
            "dither": feature_extractor.get("dither", 0.0),
            "input_scale_factor": feature_extractor.get(
                "input_scale_factor", 1.0
            ),
            "mel_floor": feature_extractor.get("mel_floor", 1e-5),
            "per_bin_mean": feature_extractor.get("per_bin_mean"),
            "per_bin_stddev": feature_extractor.get("per_bin_stddev"),
            "padding_side": feature_extractor.get("padding_side", "right"),
        }
    except Exception:
        return None


def convert_tokenizer(cls, preset, **kwargs):
    proto = get_file(preset, "tokenizer.model")
    return cls(proto=proto, **kwargs)
