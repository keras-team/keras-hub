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

from dataclasses import dataclass

import keras
from keras import KerasTensor
from keras import layers
from keras import ops


class MLPEmbedder(keras.Model):
    """
    A simple multi-layer perceptron (MLP) embedder model.

    This model applies a linear transformation followed by the SiLU activation
    function and another linear transformation to the input tensor.
    """

    def __init__(self, hidden_dim: int):
        """
        Initializes the MLPEmbedder.

        Args:
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        self.in_layer = layers.Dense(hidden_dim, use_bias=True)
        self.silu = layers.Activation("silu")
        self.out_layer = layers.Dense(hidden_dim, use_bias=True)

    def call(self, x: KerasTensor) -> KerasTensor:
        """
        Applies the MLP embedding to the input tensor.

        Args:
            x (KerasTensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            KerasTensor: Output tensor of shape (batch_size, hidden_dim) after applying
            the MLP transformations.
        """
        x = self.in_layer(x)
        x = self.silu(x)
        return self.out_layer(x)


# TODO: Maybe this can be exported as part of the public API? Seems to have enough reusability.
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value and applies
    a learned scaling factor.
    """

    def __init__(self, dim: int):
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): The dimensionality of the input tensor.
        """
        super().__init__()
        self.scale = self.add_weight(
            name="scale", shape=(dim,), initializer="ones"
        )

    def call(self, x: KerasTensor) -> KerasTensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (KerasTensor): Input tensor of shape (batch_size, dim).

        Returns:
            KerasTensor: The RMS-normalized tensor of the same shape (batch_size, dim),
            scaled by the learned `scale` parameter.
        """
        x = ops.cast(x, float)
        rrms = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6)
        return (x * rrms) * self.scale


class QKNorm(keras.layers.Layer):
    """
    A layer that applies RMS normalization to query and key tensors.

    This layer normalizes the input query and key tensors using separate RMSNorm
    layers for each.
    """

    def __init__(self, dim: int):
        """
        Initializes the QKNorm layer.

        Args:
            dim (int): The dimensionality of the input query and key tensors.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def call(
        self, q: KerasTensor, k: KerasTensor
    ) -> tuple[KerasTensor, KerasTensor]:
        """
        Applies RMS normalization to the query and key tensors.

        Args:
            q (KerasTensor): The query tensor of shape (batch_size, dim).
            k (KerasTensor): The key tensor of shape (batch_size, dim).

        Returns:
            tuple[KerasTensor, KerasTensor]: A tuple containing the normalized query and key tensors.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


@dataclass
class ModulationOut:
    shift: KerasTensor
    scale: KerasTensor
    gate: KerasTensor


class Modulation(keras.Model):
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = keras.layers.Dense(self.multiplier * dim, use_bias=True)

    def call(self, x):
        x = keras.layers.Activation("silu")(x)
        out = self.lin(x)
        out = keras.ops.split(
            out[:, None, :], indices_or_sections=self.multiplier, axis=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
