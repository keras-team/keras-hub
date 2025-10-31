import gc
import os
import types

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import Gemma3nForConditionalGeneration
from transformers import Gemma3nProcessor

from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.models.mobilenetv5.mobilenetv5_attention import (
    MobileAttention,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import EdgeResidual
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import (
    UniversalInvertedResidual,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
    convert_arch_def_to_stackwise,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct

PRESET_MAP = {
    "gemma3n_e2b": "google/gemma-3n-E2B",
}
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "cache_dir", "./hf_cache", "Directory to cache Hugging Face downloads."
)
flags.mark_flag_as_required("preset")


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


def convert_model(hf_config, dtype=None):
    text_config = hf_config.text_config
    vision_config = hf_config.vision_config
    audio_config = hf_config.audio_config
    vision_encoder_config = MODEL_CONFIGS["mobilenetv5_300m_enc"].copy()
    vision_encoder_config["image_shape"] = (768, 768, 3)
    if text_config.hidden_activation == "gelu_pytorch_tanh":
        text_config.hidden_activation = "gelu_approximate"
    gemma3n_backbone = Gemma3nBackbone(
        text_vocab_size=text_config.vocab_size,
        text_hidden_size=text_config.hidden_size,
        num_hidden_layers=text_config.num_hidden_layers,
        pad_token_id=0,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        intermediate_size=text_config.intermediate_size,
        hidden_activation=text_config.hidden_activation,
        layer_types=text_config.layer_types,
        sliding_window=text_config.sliding_window,
        rope_theta=text_config.rope_theta,
        max_position_embeddings=text_config.max_position_embeddings,
        vocab_size_per_layer_input=text_config.vocab_size_per_layer_input,
        hidden_size_per_layer_input=text_config.hidden_size_per_layer_input,
        altup_num_inputs=text_config.altup_num_inputs,
        laurel_rank=text_config.laurel_rank,
        attention_bias=text_config.attention_bias,
        attention_dropout=text_config.attention_dropout,
        rope_scaling=text_config.rope_scaling,
        activation_sparsity_pattern=text_config.activation_sparsity_pattern,
        altup_coef_clip=text_config.altup_coef_clip,
        altup_active_idx=text_config.altup_active_idx,
        altup_correct_scale=text_config.altup_correct_scale,
        num_kv_shared_layers=text_config.num_kv_shared_layers,
        vision_encoder_config=vision_encoder_config,
        vision_hidden_size=vision_config.hidden_size,
        vision_vocab_size=vision_config.vocab_size,
        vision_vocab_offset=vision_config.vocab_offset,
        vision_soft_tokens_per_image=hf_config.vision_soft_tokens_per_image,
        image_token_id=hf_config.image_token_id,
        audio_encoder_config=audio_config.to_dict(),
        audio_hidden_size=audio_config.hidden_size,
        audio_vocab_size=audio_config.vocab_size,
        audio_vocab_offset=audio_config.vocab_offset,
        audio_soft_tokens_per_image=hf_config.audio_soft_tokens_per_image,
        audio_token_id=hf_config.audio_token_id,
        rms_norm_eps=text_config.rms_norm_eps,
        dtype=dtype,
    )
    return gemma3n_backbone


class HfToKerasConverter:
    def __init__(self, hf_model):
        self.hf_state_dict = {
            k: v for k, v in hf_model.state_dict().items() if "lm_head" not in k
        }

    def _port_weights(self, layer_or_variable, hf_key, transpose_dims=None):
        if hf_key not in self.hf_state_dict:
            print(f"⚠️ Weight key not found in state_dict: {hf_key}")
            return
        weights = self.hf_state_dict[hf_key].cpu().float().numpy()
        if transpose_dims:
            weights = weights.transpose(transpose_dims)

        if hasattr(layer_or_variable, "assign"):
            layer_or_variable.assign(weights)
            return

        current_weights = layer_or_variable.get_weights()
        if (
            not current_weights
            and hasattr(layer_or_variable, "weights")
            and not layer_or_variable.weights
        ):
            print(
                f"⚠️ Keras layer {layer_or_variable.name} has no weights to "
                "set. Skipping."
            )
            return
        if len(current_weights) == 1:
            layer_or_variable.set_weights([weights])
        elif len(current_weights) == 2:
            bias_key = hf_key.replace(".weight", ".bias")
            if bias_key in self.hf_state_dict:
                bias = self.hf_state_dict[bias_key].cpu().numpy()
                layer_or_variable.set_weights([weights, bias])
            else:
                layer_or_variable.set_weights([weights, current_weights[1]])
        else:
            print(
                f"❓ Unexpected number of weights in layer "
                f"{layer_or_variable.name}"
            )

    def _port_rms_norm(self, layer, hf_prefix):
        key = f"{hf_prefix}.weight"
        self._port_weights(layer, key)

    def _port_bn(self, layer, hf_prefix):
        keys = [
            f"{hf_prefix}.weight",
            f"{hf_prefix}.bias",
            f"{hf_prefix}.running_mean",
            f"{hf_prefix}.running_var",
        ]
        weights = [
            self.hf_state_dict[key].cpu().float().numpy() for key in keys
        ]
        layer.set_weights(weights)

    def _port_cna(self, cna_layer: ConvNormAct, hf_conv_prefix, hf_norm_prefix):
        if isinstance(cna_layer.conv, keras.layers.DepthwiseConv2D):
            self._port_weights(
                cna_layer.conv,
                f"{hf_conv_prefix}.weight",
                transpose_dims=(2, 3, 0, 1),
            )
        else:
            self._port_weights(
                cna_layer.conv,
                f"{hf_conv_prefix}.weight",
                transpose_dims=(2, 3, 1, 0),
            )
        if f"{hf_norm_prefix}.running_mean" in self.hf_state_dict:
            self._port_bn(cna_layer.norm, hf_norm_prefix)
        else:
            self._port_rms_norm(cna_layer.norm, hf_norm_prefix)

    def _port_attn(self, attn_layer, hf_attn_prefix):
        self._port_weights(
            attn_layer.query_layers[-1],
            f"{hf_attn_prefix}.query.proj.weight",
            (2, 3, 1, 0),
        )
        if len(attn_layer.key_layers) > 1:
            self._port_weights(
                attn_layer.key_layers[0],
                f"{hf_attn_prefix}.key.down_conv.weight",
                (2, 3, 0, 1),
            )
            key_norm_layer = attn_layer.key_layers[1]
            if f"{hf_attn_prefix}.key.norm.running_mean" in self.hf_state_dict:
                self._port_bn(key_norm_layer, f"{hf_attn_prefix}.key.norm")
            else:
                self._port_rms_norm(
                    key_norm_layer, f"{hf_attn_prefix}.key.norm"
                )
        self._port_weights(
            attn_layer.key_layers[-1],
            f"{hf_attn_prefix}.key.proj.weight",
            (2, 3, 1, 0),
        )
        if len(attn_layer.value_layers) > 1:
            self._port_weights(
                attn_layer.value_layers[0],
                f"{hf_attn_prefix}.value.down_conv.weight",
                (2, 3, 0, 1),
            )
            value_norm_layer = attn_layer.value_layers[1]
            if (
                f"{hf_attn_prefix}.value.norm.running_mean"
                in self.hf_state_dict
            ):
                self._port_bn(value_norm_layer, f"{hf_attn_prefix}.value.norm")
            else:
                self._port_rms_norm(
                    value_norm_layer, f"{hf_attn_prefix}.value.norm"
                )
        self._port_weights(
            attn_layer.value_layers[-1],
            f"{hf_attn_prefix}.value.proj.weight",
            (2, 3, 1, 0),
        )
        self._port_weights(
            attn_layer.output_proj_layers[-2],
            f"{hf_attn_prefix}.output.proj.weight",
            (2, 3, 1, 0),
        )

    def _port_vision_tower(self, keras_model):
        print("  -> Porting vision tower (MobileNetV5)...")
        backbone = keras_model.vision_encoder
        hf_prefix = "model.vision_tower.timm_model"

        stem_layer = backbone.get_layer("conv_stem")
        self._port_cna(
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
                    self._port_cna(
                        block.conv_exp,
                        f"{block_prefix}.conv_exp",
                        f"{block_prefix}.bn1",
                    )
                    self._port_cna(
                        block.conv_pwl,
                        f"{block_prefix}.conv_pwl",
                        f"{block_prefix}.bn2",
                    )
                elif isinstance(block, UniversalInvertedResidual):
                    if hasattr(block, "dw_start") and not isinstance(
                        block.dw_start, types.FunctionType
                    ):
                        self._port_cna(
                            block.dw_start,
                            f"{block_prefix}.dw_start.conv",
                            f"{block_prefix}.dw_start.bn",
                        )
                    self._port_cna(
                        block.pw_exp,
                        f"{block_prefix}.pw_exp.conv",
                        f"{block_prefix}.pw_exp.bn",
                    )
                    if hasattr(block, "dw_mid") and not isinstance(
                        block.dw_mid, types.FunctionType
                    ):
                        self._port_cna(
                            block.dw_mid,
                            f"{block_prefix}.dw_mid.conv",
                            f"{block_prefix}.dw_mid.bn",
                        )
                    self._port_cna(
                        block.pw_proj,
                        f"{block_prefix}.pw_proj.conv",
                        f"{block_prefix}.pw_proj.bn",
                    )
                    gamma_key = f"{block_prefix}.layer_scale.gamma"
                    if gamma_key in self.hf_state_dict:
                        self._port_weights(block.layer_scale, gamma_key)
                elif isinstance(block, MobileAttention):
                    self._port_rms_norm(block.norm, f"{block_prefix}.norm")
                    gamma_key = f"{block_prefix}.layer_scale.gamma"
                    if gamma_key in self.hf_state_dict:
                        self._port_weights(block.layer_scale, gamma_key)
                    attn_prefix = f"{block_prefix}.attn"
                    self._port_attn(block.attn, attn_prefix)
                block_counter += 1
        try:
            msfa_layer = backbone.get_layer("msfa")
            msfa_prefix = f"{hf_prefix}.msfa"
            ffn = msfa_layer.ffn
            self._port_cna(
                ffn.pw_exp,
                f"{msfa_prefix}.ffn.pw_exp.conv",
                f"{msfa_prefix}.ffn.pw_exp.bn",
            )
            self._port_cna(
                ffn.pw_proj,
                f"{msfa_prefix}.ffn.pw_proj.conv",
                f"{msfa_prefix}.ffn.pw_proj.bn",
            )
            self._port_rms_norm(msfa_layer.norm, f"{msfa_prefix}.norm")
        except ValueError:
            pass

    def _port_language_model(self, keras_model):
        print("  -> Porting language model...")
        lm = keras_model.language_model
        hf_prefix = "model.language_model"

        self._port_weights(
            lm.embed_tokens.embedding, f"{hf_prefix}.embed_tokens.weight"
        )
        self._port_rms_norm(lm.norm, f"{hf_prefix}.norm")
        self._port_weights(
            lm.embed_tokens_per_layer.embedding,
            f"{hf_prefix}.embed_tokens_per_layer.weight",
        )
        self._port_weights(
            lm.per_layer_model_projection,
            f"{hf_prefix}.per_layer_model_projection.weight",
            transpose_dims=(1, 0),
        )
        self._port_rms_norm(
            lm.per_layer_projection_norm,
            f"{hf_prefix}.per_layer_projection_norm",
        )

        for i, proj in enumerate(lm.altup_projections):
            self._port_weights(
                proj,
                f"{hf_prefix}.altup_projections.{i}.weight",
                transpose_dims=(1, 0),
            )
        for i, proj in enumerate(lm.altup_unembed_projections):
            self._port_weights(
                proj,
                f"{hf_prefix}.altup_unembed_projections.{i}.weight",
                transpose_dims=(1, 0),
            )

        for i, layer in enumerate(lm.layers):
            layer_prefix = f"{hf_prefix}.layers.{i}"

            # Attention
            self._port_weights(
                layer.attention.q_proj,
                f"{layer_prefix}.self_attn.q_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.attention.k_proj,
                f"{layer_prefix}.self_attn.k_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.attention.v_proj,
                f"{layer_prefix}.self_attn.v_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.attention.o_proj,
                f"{layer_prefix}.self_attn.o_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                layer.attention.q_norm, f"{layer_prefix}.self_attn.q_norm"
            )
            self._port_rms_norm(
                layer.attention.k_norm, f"{layer_prefix}.self_attn.k_norm"
            )

            # MLP
            self._port_weights(
                layer.mlp.gate_proj,
                f"{layer_prefix}.mlp.gate_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.mlp.up_proj,
                f"{layer_prefix}.mlp.up_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.mlp.down_proj,
                f"{layer_prefix}.mlp.down_proj.weight",
                transpose_dims=(1, 0),
            )

            # LayerNorms
            self._port_rms_norm(
                layer.input_layernorm, f"{layer_prefix}.input_layernorm"
            )
            self._port_rms_norm(
                layer.post_attention_layernorm,
                f"{layer_prefix}.post_attention_layernorm",
            )
            self._port_rms_norm(
                layer.pre_feedforward_layernorm,
                f"{layer_prefix}.pre_feedforward_layernorm",
            )
            self._port_rms_norm(
                layer.post_feedforward_layernorm,
                f"{layer_prefix}.post_feedforward_layernorm",
            )

            # AltUp
            altup_prefix = f"{layer_prefix}.altup"
            self._port_weights(
                layer.altup.correction_coefs,
                f"{altup_prefix}.correction_coefs.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.altup.prediction_coefs,
                f"{altup_prefix}.prediction_coefs.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.altup.modality_router,
                f"{altup_prefix}.modality_router.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                layer.altup.router_norm, f"{altup_prefix}.router_norm"
            )
            if layer.altup.altup_correct_scale:
                self._port_weights(
                    layer.altup.correct_output_scale,
                    f"{altup_prefix}.correct_output_scale",
                )

            # Laurel
            laurel_prefix = f"{layer_prefix}.laurel"
            self._port_weights(
                layer.laurel.linear_left,
                f"{laurel_prefix}.linear_left.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.laurel.linear_right,
                f"{laurel_prefix}.linear_right.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                layer.laurel.post_laurel_norm,
                f"{laurel_prefix}.post_laurel_norm",
            )

            # Per-layer inputs
            self._port_weights(
                layer.per_layer_input_gate,
                f"{layer_prefix}.per_layer_input_gate.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                layer.per_layer_projection,
                f"{layer_prefix}.per_layer_projection.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                layer.post_per_layer_input_norm,
                f"{layer_prefix}.post_per_layer_input_norm",
            )

    def _port_audio_tower(self, keras_model):
        print("  -> Porting audio tower...")
        audio_encoder = keras_model.audio_encoder
        hf_prefix = "model.audio_tower"

        ssp = audio_encoder.subsample_conv_projection
        ssp_prefix = f"{hf_prefix}.subsample_conv_projection"
        self._port_weights(
            ssp.conv_0.conv,
            f"{ssp_prefix}.conv_0.conv.weight",
            transpose_dims=(2, 3, 1, 0),
        )
        self._port_weights(
            ssp.conv_0.norm.scale, f"{ssp_prefix}.conv_0.norm.weight"
        )
        self._port_weights(
            ssp.conv_1.conv,
            f"{ssp_prefix}.conv_1.conv.weight",
            transpose_dims=(2, 3, 1, 0),
        )
        self._port_weights(
            ssp.conv_1.norm.scale, f"{ssp_prefix}.conv_1.norm.weight"
        )
        self._port_weights(
            ssp.input_proj_linear,
            f"{ssp_prefix}.input_proj_linear.weight",
            transpose_dims=(1, 0),
        )

        for i, block in enumerate(audio_encoder.conformer):
            block_prefix = f"{hf_prefix}.conformer.{i}"
            ffw_start_prefix = f"{block_prefix}.ffw_layer_start"
            self._port_rms_norm(
                block.ffw_layer_start.pre_layer_norm,
                f"{ffw_start_prefix}.pre_layer_norm",
            )
            self._port_weights(
                block.ffw_layer_start.ffw_layer_1,
                f"{ffw_start_prefix}.ffw_layer_1.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.ffw_layer_start.ffw_layer_2,
                f"{ffw_start_prefix}.ffw_layer_2.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                block.ffw_layer_start.post_layer_norm,
                f"{ffw_start_prefix}.post_layer_norm",
            )

            attn_prefix = f"{block_prefix}.attention"
            self._port_rms_norm(
                block.attention.pre_attn_norm, f"{attn_prefix}.pre_attn_norm"
            )
            self._port_weights(
                block.attention.attn.per_dim_scale,
                f"{attn_prefix}.attn.per_dim_scale",
            )
            self._port_weights(
                block.attention.attn.relative_position_embedding.pos_proj,
                f"{attn_prefix}.attn.relative_position_embedding.pos_proj.weight",  # noqa: E501
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.attention.attn.q_proj,
                f"{attn_prefix}.attn.q_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.attention.attn.k_proj,
                f"{attn_prefix}.attn.k_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.attention.attn.v_proj,
                f"{attn_prefix}.attn.v_proj.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.attention.post,
                f"{attn_prefix}.post.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                block.attention.post_norm, f"{attn_prefix}.post_norm"
            )

            lconv_prefix = f"{block_prefix}.lconv1d"
            self._port_rms_norm(
                block.lconv1d.pre_layer_norm, f"{lconv_prefix}.pre_layer_norm"
            )
            self._port_weights(
                block.lconv1d.linear_start,
                f"{lconv_prefix}.linear_start.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.lconv1d.depthwise_conv1d,
                f"{lconv_prefix}.depthwise_conv1d.weight",
                transpose_dims=(2, 0, 1),
            )
            self._port_rms_norm(
                block.lconv1d.conv_norm, f"{lconv_prefix}.conv_norm"
            )
            self._port_weights(
                block.lconv1d.linear_end,
                f"{lconv_prefix}.linear_end.weight",
                transpose_dims=(1, 0),
            )

            ffw_end_prefix = f"{block_prefix}.ffw_layer_end"
            self._port_rms_norm(
                block.ffw_layer_end.pre_layer_norm,
                f"{ffw_end_prefix}.pre_layer_norm",
            )
            self._port_weights(
                block.ffw_layer_end.ffw_layer_1,
                f"{ffw_end_prefix}.ffw_layer_1.weight",
                transpose_dims=(1, 0),
            )
            self._port_weights(
                block.ffw_layer_end.ffw_layer_2,
                f"{ffw_end_prefix}.ffw_layer_2.weight",
                transpose_dims=(1, 0),
            )
            self._port_rms_norm(
                block.ffw_layer_end.post_layer_norm,
                f"{ffw_end_prefix}.post_layer_norm",
            )
            self._port_rms_norm(block.norm, f"{block_prefix}.norm")

    def _port_multimodal_embedders(self, keras_model):
        print("  -> Porting multimodal embedders...")
        vision_prefix = "model.embed_vision"
        self._port_weights(
            keras_model.embed_vision.embedding,
            f"{vision_prefix}.embedding.weight",
        )
        self._port_rms_norm(
            keras_model.embed_vision.hard_embedding_norm,
            f"{vision_prefix}.hard_embedding_norm",
        )
        self._port_rms_norm(
            keras_model.embed_vision.soft_embedding_norm,
            f"{vision_prefix}.soft_embedding_norm",
        )
        self._port_weights(
            keras_model.embed_vision.embedding_projection,
            f"{vision_prefix}.embedding_projection.weight",
            transpose_dims=(1, 0),
        )

        audio_prefix = "model.embed_audio"
        self._port_weights(
            keras_model.embed_audio.embedding,
            f"{audio_prefix}.embedding.weight",
        )
        self._port_rms_norm(
            keras_model.embed_audio.hard_embedding_norm,
            f"{audio_prefix}.hard_embedding_norm",
        )
        self._port_rms_norm(
            keras_model.embed_audio.soft_embedding_norm,
            f"{audio_prefix}.soft_embedding_norm",
        )
        self._port_weights(
            keras_model.embed_audio.embedding_projection,
            f"{audio_prefix}.embedding_projection.weight",
            transpose_dims=(1, 0),
        )

    def convert(self, keras_model: Gemma3nBackbone):
        print("🔶 Starting weight conversion...")
        self._port_vision_tower(keras_model)
        self._port_language_model(keras_model)
        self._port_audio_tower(keras_model)
        self._port_multimodal_embedders(keras_model)
        print("✅ Full backbone weights converted.")


def validate_output(keras_model, hf_model, hf_processor):
    print("🔶 Validating model outputs...")
    image_size = hf_processor.image_processor.size
    image = Image.new("RGB", (image_size["width"], image_size["height"]))
    sampling_rate = hf_processor.feature_extractor.sampling_rate
    audio_data = np.zeros(int(sampling_rate * 2.0))
    text = f"A cat sat on a mat{hf_processor.image_token}<end_of_turn>\n{hf_processor.audio_token}"  # noqa: E501
    hf_inputs = hf_processor(
        text=text,
        images=image,
        audio=[audio_data],
        return_tensors="pt",
        padding="longest",
    )
    print("  -> Running HF model forward pass...")
    with torch.no_grad():
        hf_output = hf_model.model(**hf_inputs).last_hidden_state
        hf_output = hf_output.detach().cpu().float().numpy()
    print(f"  -> HF model output shape: {hf_output.shape}")
    keras_inputs = {k: v.numpy() for k, v in hf_inputs.items()}
    keras_inputs["token_ids"] = keras_inputs.pop("input_ids")
    if "token_type_ids" in keras_inputs:
        del keras_inputs["token_type_ids"]
    keras_inputs["pixel_values"] = np.transpose(
        keras_inputs["pixel_values"], (0, 2, 3, 1)
    )
    if keras_inputs["pixel_values"].ndim == 4:
        keras_inputs["pixel_values"] = np.expand_dims(
            keras_inputs["pixel_values"], axis=1
        )
    input_shape = keras_inputs["token_ids"].shape
    seq_len = input_shape[1]
    attention_mask_2d = keras_inputs["attention_mask"]
    attention_mask_4d = attention_mask_2d[:, None, None, :]
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))[
        None, None, :, :
    ]
    final_mask = causal_mask & attention_mask_4d
    keras_inputs["attention_mask"] = final_mask
    print("  -> Running Keras model forward pass...")
    keras_output = keras_model.predict(keras_inputs)
    print(f"  -> Keras model output shape: {keras_output.shape}")
    mean_diff = np.mean(np.abs(keras_output - hf_output))
    print(f"🔶 Mean absolute difference: {mean_diff}")


def main(_):
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    cache_dir = FLAGS.cache_dir
    save_path = preset
    model_cache_path = os.path.join(cache_dir, f"{preset}_model")
    processor_cache_path = os.path.join(cache_dir, f"{preset}_processor")
    hf_model = None
    hf_processor = None
    if os.path.exists(model_cache_path) and os.path.exists(
        processor_cache_path
    ):
        print(
            "  -> Loading cached Hugging Face model and processor from "
            "{cache_dir}"
        )
        try:
            hf_model = Gemma3nForConditionalGeneration.from_pretrained(
                model_cache_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            hf_processor = Gemma3nProcessor.from_pretrained(
                processor_cache_path
            )
        except Exception as e:
            print(f"⚠️ Failed to load from cache: {e}. Downloading again...")
            hf_model = None
            hf_processor = None
    if hf_model is None or hf_processor is None:
        print(f"  -> Downloading Hugging Face model: {hf_model_name}")
        hf_model = Gemma3nForConditionalGeneration.from_pretrained(
            hf_model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        hf_processor = Gemma3nProcessor.from_pretrained(hf_model_name)
        print(f"💾 Saving model and processor to cache: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        hf_model.save_pretrained(model_cache_path)
        hf_processor.save_pretrained(processor_cache_path)
    hf_model.eval()
    print("-> Creating Keras model from HF config.")
    keras_model = convert_model(hf_model.config, dtype="bfloat16")
    print("-> Converting weights from HF to Keras.")
    converter = HfToKerasConverter(hf_model)
    converter.convert(keras_model)
    print("\n-> Validating output consistency.")
    validate_output(keras_model, hf_model, hf_processor)
    print(f"💾 Saving Keras preset to ./{save_path}")
    keras_model.save_to_preset(f"./{save_path}")
    print("🏁 Conversion complete.")
    del hf_model
    gc.collect()


if __name__ == "__main__":
    app.run(main)
