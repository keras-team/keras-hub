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
from keras import layers
from keras import ops


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


class CLIPEncoderBlock(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`. "
                f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation

        if intermediate_activation == "quick_gelu":
            intermediate_activation = quick_gelu

        self.layer_norm_1 = layers.LayerNormalization(
            epsilon=0.00001, dtype=self.dtype_policy, name="layer_norm_1"
        )
        self.attention = layers.MultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm_2 = layers.LayerNormalization(
            epsilon=0.00001, dtype=self.dtype_policy, name="layer_norm_2"
        )
        self.dense_1 = layers.Dense(
            self.intermediate_dim, dtype=self.dtype_policy, name="dense_1"
        )
        self.activation = layers.Activation(
            intermediate_activation, dtype=self.dtype_policy, name="activation"
        )
        self.dense_2 = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="dense_2"
        )

    def build(self, input_shape):
        self.layer_norm_1.build(input_shape)
        self.attention.build(input_shape, input_shape, input_shape)
        self.layer_norm_2.build(input_shape)
        self.dense_1.build(input_shape)
        input_shape = self.dense_1.compute_output_shape(input_shape)
        self.dense_2.build(input_shape)

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape

    def call(self, x, training=None):
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(x, x, x, training=training, use_causal_mask=True)
        x = ops.add(residual, x)

        residual = x
        x = self.dense_1(self.layer_norm_2(residual))
        x = self.activation(x)
        x = self.dense_2(x)
        x = ops.add(residual, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
            }
        )
        return config
