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

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.AlibiBias")
class AlibiBias(keras.layers.Layer):
    """A layer that adds the alibi bias to attention scores.

    This layer adds the alibi bias to the attention scores. Alibi bias is a
    linear, non-learned bias. Defined and formalized in
    [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).

    This layer takes as input the attention scores. and returns the attention
    scores after adding the alibi bias to it. The output will have the same
    shape as the input.

    Args:
        alibi_bias_max: int. This value will be used to compute the slope of
            each head. The heads' slopes are a geometric sequence that starts at
            `2**(-alibi_bias_max/num_heads)` and uses that same value as its
            ratio. Defaults to 8.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        attention_scores: The result of multipying the query and the key of the
            multi-head attention layer of the transformer to add alibi bias to
            it. With shape `(batch_size, num_heads, query_length, key_length)`.

    Example:
    ```python
    query_length = 10
    key_length = 10
    num_heads = 4
    batch_size = 2
    hidden_dim = 8

    # Create new alibi layer.
    alibi_layer = keras_hub.layers.AlibiBias()

    query = np.zeros((batch_size, num_heads, query_length, hidden_dim))
    key = np.zeros((batch_size, num_heads, hidden_dim, key_length))

    attention_scores = keras.ops.matmul(query, key)

    # Add alibi bias to attention scores.
    attention_scores = alibi_layer(attention_scores)
    ```

    References:
     - [Press et al., 2021](https://arxiv.org/abs/2108.12409)
    """

    def __init__(
        self,
        alibi_bias_max=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alibi_bias_max = alibi_bias_max

    def call(self, attention_scores):
        shape = ops.shape(attention_scores)
        if len(shape) != 4:
            raise ValueError(
                "Expected `attention_scores` shape to be "
                "`(batch_size, num_heads, query_length, key_Length)`."
                f" Recived shape={shape}"
            )

        key_length = shape[-1]
        num_heads = shape[-3]

        alibi_bias = self._get_alibi_bias(num_heads, key_length)

        return ops.add(attention_scores, alibi_bias)

    def _get_alibi_bias(self, num_heads, key_length):
        slopes = ops.convert_to_tensor(
            self._get_slopes(num_heads), dtype=self.compute_dtype
        )
        slopes = ops.expand_dims(slopes, 1)

        seq_range = ops.expand_dims(
            ops.arange(1 - key_length, 1, dtype="int32"), 0
        )
        seq_range = ops.cast(seq_range, dtype=self.compute_dtype)

        alibi_bias = ops.multiply(slopes, seq_range)
        alibi_bias = ops.expand_dims(alibi_bias, 1)

        # return shape is `(1, num_heads, 1, key_length)`
        return ops.expand_dims(alibi_bias, 0)

    def _get_slopes(self, num_heads):
        # this function is adopted from Alibi original implementation.
        # https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        def get_slopes_power_of_2(n):
            start = 2 ** (
                -(2 ** -(math.log2(n) - math.log2(self.alibi_bias_max)))
            )
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : num_heads - closest_power_of_2
                ]
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alibi_bias_max": self.alibi_bias_max,
            }
        )
        return config
