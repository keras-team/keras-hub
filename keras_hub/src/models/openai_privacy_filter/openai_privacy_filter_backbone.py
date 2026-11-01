"""OpenAI Privacy Filter encoder-only backbone.

An encoder-only MoE transformer with bidirectional sliding-window attention,
interleaved YaRN RoPE, attention sinks, and GLU experts.

References:
- HF: transformers/models/openai_privacy_filter/modular_openai_privacy_filter.py
- Parent: transformers/models/gpt_oss/modeling_gpt_oss.py
"""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gpt_oss.gpt_oss_layer_norm import (
    GptOssLayerNormalization,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_layers import (  # noqa: E501
    OpenAIPrivacyFilterEncoderLayer,
)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
@keras_hub_export("keras_hub.models.OpenAIPrivacyFilterBackbone")
class OpenAIPrivacyFilterBackbone(Backbone):
    """OpenAI Privacy Filter encoder-only backbone.

    A compact (~1.4B param) encoder-only MoE transformer for PII detection.
    Uses bidirectional sliding-window attention with sinks, interleaved YaRN
    RoPE, and GLU experts.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        num_layers: int. Number of encoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of KV heads (GQA).
        hidden_dim: int. Hidden dimension size.
        intermediate_dim: int. FFN intermediate dimension.
        head_dim: int. Per-head dimension. Defaults to `64`.
        num_experts: int. Number of MoE experts. Defaults to `128`.
        top_k: int. Number of active experts per token. Defaults to `4`.
        rope_max_wavelength: float. RoPE base theta. Defaults to `150000.0`.
        rope_scaling_factor: float. YaRN scaling factor. Defaults to `1.0`.
        rope_type: string. RoPE type. Defaults to `"yarn"`.
        beta_fast: float. YaRN beta fast parameter. Defaults to `32.0`.
        beta_slow: float. YaRN beta slow parameter. Defaults to `1.0`.
        original_max_position_embeddings: int. Defaults to `4096`.
        truncate: bool. Whether to truncate RoPE values. Defaults to `False`.
        swiglu_alpha: float. SwiGLU alpha parameter. Defaults to `1.702`.
        swiglu_limit: float. SwiGLU limit parameter. Defaults to `7.0`.
        rms_norm_eps: float. RMSNorm epsilon. Defaults to `1e-5`.
        sliding_window: int. Bidirectional sliding window size.
            Defaults to `128`.
        attention_dropout: float. Attention dropout. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    model = keras_hub.models.OpenAIPrivacyFilterBackbone.from_preset("hf://openai/privacy-filter")

    model = keras_hub.models.OpenAIPrivacyFilterBackbone(
        vocabulary_size=200064,
        num_layers=8,
        num_query_heads=14,
        num_key_value_heads=2,
        hidden_dim=640,
        intermediate_dim=640,
    )
    input_data = {
        "token_ids": np.ones((1, 12), dtype="int32"),
        "padding_mask": np.ones((1, 12), dtype="int32"),
    }
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim=64,
        num_experts=128,
        top_k=4,
        rope_max_wavelength=150000.0,
        rope_scaling_factor=1.0,
        rope_type="yarn",
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
        truncate=False,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        rms_norm_eps=1e-5,
        sliding_window=128,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="token_embedding",
        )
        self.encoder_layers = []
        for i in range(num_layers):
            layer = OpenAIPrivacyFilterEncoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                top_k=top_k,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rope_type=rope_type,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                original_max_position_embeddings=original_max_position_embeddings,
                truncate=truncate,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                rms_norm_eps=rms_norm_eps,
                sliding_window=sliding_window,
                attention_dropout=attention_dropout,
                dtype=dtype,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)
        self.final_layernorm = GptOssLayerNormalization(
            epsilon=rms_norm_eps,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)

        # Build padding attention mask: (batch, 1, seq, seq)
        # True where both query and key positions are non-padding
        pm = ops.cast(padding_mask_input, "bool")
        attn_mask = ops.logical_and(pm[:, None, :], pm[:, :, None])[
            :, None, :, :
        ]

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask=attn_mask)

        x = self.final_layernorm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_type = rope_type
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings
        self.truncate = truncate
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_type": self.rope_type,
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "original_max_position_embeddings": (
                    self.original_max_position_embeddings
                ),
                "truncate": self.truncate,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
                "rms_norm_eps": self.rms_norm_eps,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config
