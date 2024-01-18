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
    """A layer that generates alibi bias

    This layer generates a linear, non-learned bias. Defined and formalized in 
    [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).

    Takes as input an embedded token tensor. The input must have shape
    `(batch_size, sequence_length, hidden_dim)`. This layer will return an alibi
    bias of the shape `(1, num_heads, 1, sequence_length)`, which will be added to
    the result of the query-key dot product in the multi-head attention layer of
    the transformer. 

    Args:
        num_heads: int. The number of heads in the multi-head attention layer of
            the transformer.
        alibi_bias_max: int. This value will be used to compute the slope of 
            each head. The heads slopes is a geometric sequence that starts at 
            `2**(-alibi_bias_max/num_heads)` and uses that same value as its 
            ratio. Defaults to 8.
        full: bool. Whether to return the full alibi bias tensor. If set to 
            `True`, the alibi bias shape will be 
            `(1, num_heads, sequence_length, sequence_length)`. Defaults to 
            `False`, so the third dimension will be broadcasted, and this will 
            work because of the translation invariance property of the softmax, 
            let `L` be a tensor and `x` a constant, `softmax(L+x) = softmax(L)`
        batched: bool. Whether to return the alibi bias tensor with first 
            dimention equal to `batch_size`. If set to `True` the alibi bias 
            shape wil be `(batch_size, num_heads, 1, sequence_length)`. Defaults 
            to `False`, so the first dimension will be broadcasted.
    Call arguments:
        inputs: The tensor inputs to compute an embedding for, with shape
            `(batch_size, sequence_length, hidden_dim)`.

    Examples:
    ```python
    # create a simple embedding layer with sinusoidal positional encoding
    seq_len = 100
    vocab_size = 1000
    embedding_dim = 32
    inputs = keras.Input((seq_len,), dtype="float32")
    embedding = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs)
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
    outputs = embedding + positional_encoding
    ```

    References:
     - [Press et al., 2021](https://arxiv.org/abs/2108.12409)
    """

    def __init__(
        self,
        num_heads, 
        alibi_bias_max=8,
        full=False,
        batched=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.alibi_bias_max = alibi_bias_max
        self.full = full
        self.batched = batched

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        seq_length = shape[1]

        slopes = ops.convert_to_tensor(self._get_slopes(), dtype=float)
        slopes = ops.reshape(slopes, (self.num_heads, 1, 1))


        sequence_range = ops.expand_dims(ops.arange( 1 - seq_length, 1, dtype=float), 0)
        if self.full:
            sequence_range = ops.subtract(sequence_range, ops.expand_dims(ops.arange( 1 - seq_length, 1, dtype=float), 1))
            sequence_range = ops.multiply(ops.abs(sequence_range), -1)
        
        alibi_bias = slopes * ops.expand_dims(ops.arange(seq_length, dtype=float), 0)
        alibi_bias = ops.expand_dims(alibi_bias, 0)
        if self.batched:
            return ops.repeat(alibi_bias, batch_size, axis=0)

        return alibi_bias
    
    def _get_slopes(self):
        # this function is adopted from Alibi original implementation
        # https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        def get_slopes_power_of_2(n):
            start = 2 ** (
                -(2 ** -(math.log2(n) - math.log2(self.alibi_bias_max)))
            )
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(self.num_heads).is_integer():
            return get_slopes_power_of_2(self.num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(self.num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : self.num_heads - closest_power_of_2
                ]
            )

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        output_shape = [1, self.num_heads, 1, seq_length]
        if self.full:
            output_shape[2] = seq_length
        if self.batched:
            output_shape[0] = batch_size

        return tuple(output_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "alibi_bias_max": self.alibi_bias_max,
                "full": self.full,
                "batched": self.batched,
            }
        )
        return config