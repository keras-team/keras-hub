import math

import keras

from keras_hub.src.models.gemma3n.gemma3n_attention import Gemma3nTextAttention
from keras_hub.src.models.gemma3n.gemma3n_text_layers import Gemma3nTextAltUp
from keras_hub.src.models.gemma3n.gemma3n_text_layers import (
    Gemma3nTextLaurelBlock,
)
from keras_hub.src.models.gemma3n.gemma3n_text_layers import Gemma3nTextMLP
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nTextDecoderBlock(keras.layers.Layer):
    """A layer that implements a single Gemma3n decoder block.

    This layer combines self-attention, feed-forward networks, and normalization
    to process sequences. It includes specialized components like AltUp and
    Laurel blocks for enhanced performance.

    Args:
        hidden_size: int. The size of the hidden states.
        rms_norm_eps: float. The epsilon value for the Gemma 3n RMS
            normalization layers.
        num_attention_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key and value heads for
            Grouped-Query Attention.
        head_dim: int. The dimension of each attention head.
        attention_bias: bool. If `True`, attention layers will use a bias.
        attention_dropout: float. The dropout rate for the attention mechanism.
        is_sliding: bool. If `True`, enables sliding window attention.
        sliding_window: int. The size of the sliding window for attention.
        intermediate_size: int. The size of the intermediate layer in the MLP.
        hidden_activation: str. The activation function for the MLP.
        activation_sparsity: float. Sparsity factor for the activation function.
        altup_num_inputs: int. The number of inputs for the AltUp layer.
        altup_coef_clip: float. Coefficient clipping value for the AltUp layer.
        altup_active_idx: int. The index of the active prediction in the
            AltUp layer.
        altup_correct_scale: bool. Whether to scale the corrected output from
            the AltUp layer.
        laurel_rank: int. The rank for the Laurel block.
        hidden_size_per_layer_input: int. The hidden size for the per-layer
            input projection.
    """

    def __init__(
        self,
        hidden_size,
        rms_norm_eps,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        attention_bias,
        attention_dropout,
        is_sliding,
        sliding_window,
        intermediate_size,
        hidden_activation,
        activation_sparsity,
        altup_num_inputs,
        altup_coef_clip,
        altup_active_idx,
        altup_correct_scale,
        laurel_rank,
        hidden_size_per_layer_input,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.activation_sparsity = activation_sparsity
        self.altup_num_inputs = altup_num_inputs
        self.altup_coef_clip = altup_coef_clip
        self.altup_active_idx = altup_active_idx
        self.altup_correct_scale = altup_correct_scale
        self.laurel_rank = laurel_rank
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.attention = Gemma3nTextAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window if is_sliding else None,
            name="attention",
            dtype=self.dtype_policy,
        )
        self.mlp = Gemma3nTextMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation=hidden_activation,
            activation_sparsity=activation_sparsity,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self.input_layernorm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="input_layernorm",
            dtype=self.dtype_policy,
        )
        self.post_attention_layernorm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="post_attention_layernorm",
            dtype=self.dtype_policy,
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="pre_feedforward_layernorm",
            dtype=self.dtype_policy,
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="post_feedforward_layernorm",
            dtype=self.dtype_policy,
        )
        self.altup = Gemma3nTextAltUp(
            hidden_size=hidden_size,
            altup_num_inputs=altup_num_inputs,
            altup_coef_clip=altup_coef_clip,
            altup_active_idx=altup_active_idx,
            rms_norm_eps=rms_norm_eps,
            altup_correct_scale=altup_correct_scale,
            name="altup",
            dtype=self.dtype_policy,
        )
        self.laurel = Gemma3nTextLaurelBlock(
            hidden_size=hidden_size,
            laurel_rank=laurel_rank,
            rms_norm_eps=rms_norm_eps,
            name="laurel",
            dtype=self.dtype_policy,
        )
        self.per_layer_input_gate = keras.layers.Dense(
            hidden_size_per_layer_input,
            use_bias=False,
            name="per_layer_input_gate",
            dtype=self.dtype_policy,
        )
        self.per_layer_projection = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="per_layer_projection",
            dtype=self.dtype_policy,
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="post_per_layer_input_norm",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        (
            hidden_states_shape,
            _,
            _,
            per_layer_input_shape,
            _,
        ) = input_shape
        active_prediction_shape = hidden_states_shape[1:]
        self.input_layernorm.build(active_prediction_shape)
        self.laurel.build(active_prediction_shape)
        self.attention.build(active_prediction_shape)
        self.post_attention_layernorm.build(active_prediction_shape)
        self.pre_feedforward_layernorm.build(active_prediction_shape)
        self.mlp.build(active_prediction_shape)
        self.post_feedforward_layernorm.build(active_prediction_shape)
        self.altup.build(hidden_states_shape)
        self.per_layer_input_gate.build(active_prediction_shape)
        self.per_layer_projection.build(per_layer_input_shape)
        self.post_per_layer_input_norm.build(active_prediction_shape)
        if self.hidden_activation == "gelu_approximate":
            # NOTE: `gelu_pytorch_tanh` is the same as `gelu(approximate=True)`.
            self.act_fn = lambda x: keras.activations.gelu(x, approximate=True)
        else:
            self.act_fn = keras.activations.get(self.hidden_activation)
        super().build(input_shape)

    def call(
        self,
        inputs,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
        training=False,
    ):
        (
            hidden_states,
            position_embeddings_global,
            position_embeddings_local,
            per_layer_input,
            attention_mask,
        ) = inputs
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.altup_active_idx]
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)
        position_embeddings = (
            position_embeddings_local
            if self.is_sliding
            else position_embeddings_global
        )
        if cache is not None:
            attn, _, new_cache = self.attention(
                active_prediction_normed,
                position_embeddings,
                attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=cache_update_mask,
                training=training,
            )
        else:
            attn, _ = self.attention(
                active_prediction_normed,
                position_embeddings,
                attention_mask,
                training=training,
            )
        attn = self.post_attention_layernorm(attn)
        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(
            predictions, attn_ffw_laurel_gated
        )
        corrected_predictions_list = [
            corrected_predictions[i]
            for i in range(corrected_predictions.shape[0])
        ]
        first_prediction = corrected_predictions_list[self.altup_active_idx]
        if self.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(
                first_prediction
            )
        first_prediction_gated = self.per_layer_input_gate(first_prediction)
        first_prediction_activated = self.act_fn(first_prediction_gated)
        first_prediction_multiplied = (
            first_prediction_activated * per_layer_input
        )
        first_prediction_projected = self.per_layer_projection(
            first_prediction_multiplied
        )
        first_prediction_normed = self.post_per_layer_input_norm(
            first_prediction_projected
        )
        for i in range(1, len(corrected_predictions_list)):
            corrected_predictions_list[i] = (
                corrected_predictions_list[i] + first_prediction_normed
            )
        output = keras.ops.stack(corrected_predictions_list, axis=0)
        if cache is not None:
            return output, new_cache
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "is_sliding": self.is_sliding,
                "sliding_window": self.sliding_window,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "activation_sparsity": self.activation_sparsity,
                "altup_num_inputs": self.altup_num_inputs,
                "altup_coef_clip": self.altup_coef_clip,
                "altup_active_idx": self.altup_active_idx,
                "altup_correct_scale": self.altup_correct_scale,
                "laurel_rank": self.laurel_rank,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
            }
        )
        return config
