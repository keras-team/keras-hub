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

from keras_hub.src.models.deberta_v3.disentangled_self_attention import (
    DisentangledSelfAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer

from keras_hub.src.layers.modeling.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)


class DisentangledAttentionEncoder(keras.layers.Layer):
    """Disentangled attention encoder.

    This class follows the architecture of the disentangled attention encoder
    layer in the paper
    ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    Users can instantiate multiple instances of this class to stack up a
    an encoder model which has disentangled self-attention.

    `DisentangledAttentionEncoder` is similar to
    `keras_hub.layers.TransformerEncoder`, except for the attention layer - it
    uses disentangled self-attention instead of multi-head attention.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the attention layer.
        max_position_embeddings: int. The maximum input
            sequence length. Defaults to `512`.
        bucket_size: int. The size of the relative position
            buckets. Generally equal to `max_sequence_length // 2`.
            Defaults to `256`.
        dropout: float. The dropout value, shared by
            the attention layer and feedforward network.
            Defaults to `0.0`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and disentangled
            self-attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and disentangled
            self-attention layers. Defaults to `"zeros"`.
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        max_position_embeddings=512,
        bucket_size=256,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.bucket_size = bucket_size
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True

    def build(self, inputs_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]

        # Self attention layers.
        self._self_attention_layer = DisentangledSelfAttention(
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            max_position_embeddings=self.max_position_embeddings,
            bucket_size=self.bucket_size,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        self._self_attention_layer.build(inputs_shape)
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(inputs_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(
        self,
        inputs,
        rel_embeddings,
        padding_mask=None,
        attention_mask=None,
    ):
        """Forward pass of `DisentangledAttentionEncoder`.

        Args:
            inputs: a Tensor. The input data to `DisentangledAttentionEncoder`, should be
                of shape [batch_size, sequence_length, hidden_dim].
            rel_embeddings: a Tensor. The relative position embedding matrix,
                should be of shape `[batch_size, 2 * bucket_size, hidden_dim]`.
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
                False means the certain token is masked out.
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        x = self._self_attention_layer(
            x,
            rel_embeddings=rel_embeddings,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        x = self._feedforward_layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "max_position_embeddings": self.max_position_embeddings,
                "bucket_size": self.bucket_size,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape
