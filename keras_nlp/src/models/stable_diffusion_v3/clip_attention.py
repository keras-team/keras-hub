# Copyright 2024 The KerasNLP Authors
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


class CLIPAttention(layers.Layer):
    def __init__(self, num_heads, hidden_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by num_heads. "
                f"Received: num_heads={num_heads}, hidden_dim={hidden_dim}"
            )
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.head_dim = self.hidden_dim // self.num_heads

        self.dropout_layer = layers.Dropout(self.dropout)
        self.scale = self.head_dim**-0.5
        self.query_dense = layers.Dense(
            units=self.hidden_dim, dtype=self.dtype_policy, name="query"
        )
        self.key_dense = layers.Dense(
            units=self.hidden_dim, dtype=self.dtype_policy, name="key"
        )
        self.value_dense = layers.Dense(
            units=self.hidden_dim, dtype=self.dtype_policy, name="value"
        )
        self.softmax = layers.Softmax(dtype="float32")
        self.output_dense = layers.Dense(
            units=self.hidden_dim,
            dtype=self.dtype_policy,
            name="attention_output",
        )

    def build(self, input_shape):
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.output_dense.build([None, None, self.hidden_dim])

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape

    def _transpose_for_scores(self, inputs):
        batch_size = ops.shape(inputs)[0]
        inputs = ops.reshape(
            inputs, (batch_size, -1, self.num_heads, self.head_dim)
        )
        return ops.transpose(inputs, axes=[0, 2, 1, 3])

    def call(self, x, attention_mask=None, training=None):
        batch_size = ops.shape(x)[0]
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)

        attention_logits = ops.matmul(
            query, ops.transpose(key, axes=[0, 1, 3, 2])
        )
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_logits.dtype)
        attention_logits = ops.divide(attention_logits, dk)

        if attention_mask is not None:
            attention_logits = ops.add(attention_logits, attention_mask)

        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits)
        attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        attention_output = ops.matmul(attention_softmax, value)
        attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
        attention_output = ops.reshape(
            attention_output, (batch_size, -1, self.hidden_dim)
        )
        attention_output = self.output_dense(attention_output)
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            }
        )
        return config
