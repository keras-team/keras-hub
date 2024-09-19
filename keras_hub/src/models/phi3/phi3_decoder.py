# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.phi3.phi3_attention import Phi3Attention
from keras_hub.src.models.phi3.phi3_layernorm import Phi3LayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


class Phi3Decoder(keras.layers.Layer):
    """A Transformer decoder layer for the Phi-3 backbone."""

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        activation="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        dropout=0,
        max_sequence_length=4096,
        pretraining_sequence_length=4096,
        rope_max_wavelength=10000,
        rope_scaling_type=None,
        rope_scaling_short_factor=None,
        rope_scaling_long_factor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_short_factor = rope_scaling_short_factor
        self.rope_scaling_long_factor = rope_scaling_long_factor

        self.dropout = dropout

        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, decoder_sequence_shape):

        # Pre-attention layernorm.
        self.pre_attention_layernorm = Phi3LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_layernorm",
        )
        self.pre_attention_layernorm.build(decoder_sequence_shape)

        # Self attention layer.
        self.attention = Phi3Attention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            max_sequence_length=self.max_sequence_length,
            pretraining_sequence_length=self.pretraining_sequence_length,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_type=self.rope_scaling_type,
            rope_scaling_short_factor=self.rope_scaling_short_factor,
            rope_scaling_long_factor=self.rope_scaling_long_factor,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.attention.build(decoder_sequence_shape)

        # Post-attention layernorm.
        self.post_attention_layernorm = Phi3LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(decoder_sequence_shape)

        # feedforward layers.
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(decoder_sequence_shape)

        self.feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self.feedforward_gate_dense.build(decoder_sequence_shape)

        self.feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )

        self.feedforward_output_dense.build(
            self.feedforward_gate_dense.compute_output_shape(
                decoder_sequence_shape
            )
        )

        # Dropout
        self.attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )
        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        attention_cache=None,
        attention_cache_update_index=None,
    ):
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            attention_cache=attention_cache,
            attention_cache_update_index=attention_cache_update_index,
        )
        residual = decoder_sequence
        x = self.pre_attention_layernorm(decoder_sequence)
        x = self.attention(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=attention_cache,
            cache_update_index=attention_cache_update_index,
        )
        if attention_cache is not None:
            x, attention_cache = x
        x = self.attention_dropout(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        # Note that we run the activation function in full 32-bit
        # precision since this is what `torch.nn.functional.silu`
        # does. Internally, `torch.nn.functional.silu` converts the
        # inputs to float32, computes SiLU, and converts the outputs
        # back to compute dtype.
        # CPU Kernel: https://github.com/pytorch/pytorch/blob/35c493f2cf9b623bfdc7e6b34dc1cb39690a7919/aten/src/ATen/native/cpu/Activation.cpp#L1221-L1235  # noqa: E501
        # CUDA Kernel: https://github.com/pytorch/pytorch/blob/35c493f2cf9b623bfdc7e6b34dc1cb39690a7919/aten/src/ATen/native/cuda/ActivationSiluKernel.cu  # noqa: E501
        gate_output = self.feedforward_gate_dense(x)
        gate_output = ops.cast(gate_output, "float32")
        gate_output = self.activation(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)
        x = self.feedforward_intermediate_dense(x)
        x = self.feedforward_output_dense(ops.multiply(x, gate_output))
        x = self.feedforward_dropout(x)
        decoder_output = x + residual

        if attention_cache is not None:
            return decoder_output, attention_cache
        return decoder_output

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        attention_cache,
        attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        if attention_cache is not None:
            input_length = ops.shape(attention_cache)[2]

        cache_update_index = (
            0
            if attention_cache_update_index is None
            else attention_cache_update_index
        )

        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "pretraining_sequence_length": self.pretraining_sequence_length,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_type": self.rope_scaling_type,
                "rope_scaling_short_factor": self.rope_scaling_short_factor,
                "rope_scaling_long_factor": self.rope_scaling_long_factor,
            }
        )
        return config
