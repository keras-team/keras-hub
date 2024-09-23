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

import math
from keras import ops


def timestep_embedding(
    t, dim: int, max_period=10000, time_factor: float = 1000.0
):
    """
    Creates sinusoidal timestep embeddings.

    Args:
        t (keras.Tensor): A 1-D tensor of shape (N,), representing N indices, one per batch element.
            These values may be fractional.
        dim (int): The dimension of the output.
        max_period (int, optional): Controls the minimum frequency of the embeddings. Defaults to 10000.
        time_factor (float, optional): A scaling factor applied to `t`. Defaults to 1000.0.

    Returns:
        keras.Tensor: A tensor of shape (N, D) representing the positional embeddings,
            where N is the number of batch elements and D is the specified dimension `dim`.
    """

    t = time_factor * t
    half = dim // 2
    freqs = ops.exp(
        -math.log(max_period) * ops.arange(0, half, dtype=float) / half
    )

    args = t[:, None] * freqs[None]
    embedding = ops.concatenate([ops.cos(args), ops.sin(args)], axis=-1)

    if dim % 2:
        embedding = ops.concatenate(
            [embedding, ops.zeros_like(embedding[:, :1])], axis=-1
        )

    return embedding
