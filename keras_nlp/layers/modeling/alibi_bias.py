# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops


@keras_nlp_export("keras_nlp.layers.AlibiBias")
class AlibiBias(keras.layers.Layer):
    """A layer that add the alibi bias to attention scores

    This layer generates a linear, non-learned bias. Defined and formalized in
    [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).

    Takes as input an attention score. This layer will return the attention
    scores after adding the alibi bias to it. The output will have the same
    shape as the input.

    Args:
        alibi_bias_max: int. This value will be used to compute the slope of
            each head. The heads slopes is a geometric sequence that starts at
            `2**(-alibi_bias_max/num_heads)` and uses that same value as its
            ratio. Defaults to 8.
    Call arguments:
        attention_scores: The result of multipying the query and the key of the
            multi head attention of the transformer. The shape must be greater
            than or equal to 3 with the last 3 dimensions equal to
            `(num_heads, query_length, key_length)`.

    Examples:
    ```python
    # create a simple layer that takes token embeddings as input and generates
    # the alibi tensor
    seq_len = 100
    vocab_size = 1000
    embedding_dim = 32
    inputs = keras.Input((seq_len,), dtype="float32")
    embedding = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs)
    alibi_bias = keras_nlp.layers.AlibiBias(num_heads=8)(embedding)
    ```

    References:
     - [Press et al., 2021](https://arxiv.org/abs/2108.12409)
    """

    def __init__(
        self,
        max_sequence_length,
        alibi_bias_max=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.alibi_bias_max = alibi_bias_max

    def build(self, inputs_shape):
        if len(inputs_shape) < 3:
            raise ValueError(
                "Expected `attention_scores` shape to be "
                "`(..., num_heads, query_length, key_length)`."
                f" Received shape={inputs_shape}"
            )
        num_heads = inputs_shape[-3]
        alibi_bias_shape = tuple([1 for _ in range(len(inputs_shape[:-3]))]) + (
            num_heads,
            1,
            self.max_sequence_length,
        )
        self.alibi_bias = self.add_weight(
            shape=alibi_bias_shape, dtype=self.compute_dtype, trainable=False
        )
        alibi_bias = self._get_alibi_bias(num_heads, self.max_sequence_length)
        alibi_bias = ops.reshape(
            alibi_bias,
            alibi_bias_shape,
        )

        self.alibi_bias.assign(alibi_bias)

    def call(self, attention_scores):
        shape = ops.shape(attention_scores)
        if len(shape) < 3:
            raise ValueError(
                "Expected `attention_scores` shape to be "
                "`(..., num_heads, query_length, key_length)`."
                f" Received shape={shape}"
            )

        key_length = shape[-1]

        return ops.add(
            attention_scores,
            self.alibi_bias[..., self.max_sequence_length - key_length :],
        )

    def _get_alibi_bias(self, num_heads, key_length):
        slopes = ops.convert_to_tensor(
            self._get_slopes(num_heads), dtype=self.compute_dtype
        )
        slopes = ops.expand_dims(slopes, 1)

        seq_range = ops.expand_dims(ops.arange(1 - key_length, 1), 0)
        seq_range = ops.cast(seq_range, dtype=self.compute_dtype)

        alibi_bias = ops.multiply(slopes, seq_range)

        # Expand on query dimension
        # return shape is `(num_heads, 1, key_length)`
        return ops.expand_dims(alibi_bias, 1)

    def _get_slopes(self, num_heads):
        # this function is adopted from Alibi original implementation
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
                "max_sequence_length": self.max_sequence_length,
                "alibi_bias_max": self.alibi_bias_max,
            }
        )
        return config
