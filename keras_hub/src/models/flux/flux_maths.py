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
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
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
