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
import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice


def shape_list(tensor):
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class T5MultiHeadAttention(keras.layers.Layer):
    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        num_heads,
        dropout,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.hidden_dim = hidden_dim
        self.key_value_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.use_relative_attention_bias = use_relative_attention_bias

        self.inner_dim = self.num_heads * self.key_value_dim
        self.relative_attention_buckets = 32
        self.relative_attention_max_distance = 128

        self.query_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="query_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=(self.inner_dim * self.key_value_dim) ** -0.5
            ),
        )
        self.key_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="key_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.value_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="value_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.output_projector = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            name="output_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.dropout_layer = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        if self.use_relative_attention_bias:
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_buckets, self.num_heads],
                initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.inner_dim**-0.5
                ),
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Adapted from Mesh Tensorflow.

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative
        positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute
        relative_positions. All relative positions >= max_distance map to
        the same bucket. All relative positions <= -max_distance map to
        the same bucket. This should allow for more graceful generalization to
        longer sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(
                    tf.math.greater(relative_position, 0),
                    dtype=relative_position.dtype,
                )
                * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(
                tf.cast(relative_position, tf.float32)
                / tf.cast(max_exact, tf.float32)
            )
            / tf.math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(
            relative_position_if_large, num_buckets - 1
        )
        relative_buckets += tf.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = tf.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        training=False,
    ):
        # Input is (batch_size, query_length, dim)
        # past_key_value[0] is (batch_size, num_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = shape_list(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"Argument `past_key_value` should have 2 past states: "
                    f"keys and values. Got {len(past_key_value)} past states."
                )
            real_seq_length += (
                shape_list(past_key_value[0])[2]
                if query_length is None
                else query_length
            )

        key_length = (
            real_seq_length
            if key_value_states is None
            else shape_list(key_value_states)[1]
        )

        def shape(hidden_states):
            return tf.transpose(
                tf.reshape(
                    hidden_states,
                    (batch_size, -1, self.num_heads, self.key_value_dim),
                ),
                perm=(0, 2, 1, 3),
            )

        def unshape(hidden_states):
            return tf.reshape(
                tf.transpose(hidden_states, perm=(0, 2, 1, 3)),
                (batch_size, -1, self.inner_dim),
            )

        def project(
            hidden_states, proj_layer, key_value_states, past_key_value
        ):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attention
                    # (batch_size, num_heads, key_length, dim_per_head)
                    hidden_states = tf.concat(
                        [past_key_value, hidden_states], axis=2
                    )
                else:
                    # cross-attention
                    hidden_states = past_key_value
            return hidden_states

        # get query
        query_states = shape(
            self.query_projector(hidden_states)
        )  # (batch_size, num_heads, query_length, dim_per_head)

        # get key/value
        key_states = project(
            hidden_states,
            self.key_projector,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.value_projector,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        scores = tf.einsum(
            "bnqd,bnkd->bnqk", query_states, key_states
        )  # (batch_size, num_heads, query_length, key_length)

        if position_bias is None:
            if not self.use_relative_attention_bias:
                position_bias = tf.zeros(
                    (1, self.num_heads, real_seq_length, key_length),
                    self.compute_dtype,
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated we want only
            # the last query position bias
            if past_key_value is not None:
                if not self.use_relative_attention_bias:
                    position_bias = position_bias[:, :, -seq_length:, :]
                else:
                    # we might have a padded past structure,
                    # in which case we want to fetch the position bias slice
                    # right after the most recently filled past index
                    most_recently_filled_past_index = tf.reduce_max(
                        tf.where(past_key_value[0][0, 0, :, 0] != 0.0)
                    )
                    position_bias = dynamic_slice(
                        position_bias,
                        (0, 0, most_recently_filled_past_index + 1, 0),
                        (1, self.num_heads, seq_length, real_seq_length),
                    )

            if mask is not None:
                # Add a new mask axis for the head dim.
                mask = mask[:, tf.newaxis, :, :]
                # Add a very large negative position bias for masked positions.
                mask = (1.0 - tf.cast(mask, position_bias.dtype)) * -1e9
                position_bias = position_bias + mask

        scores += position_bias
        weights = tf.nn.softmax(
            scores, axis=-1
        )  # (batch_size, num_heads, query_length, key_length)
        weights = self.dropout_layer(
            weights, training=training
        )  # (batch_size, num_heads, query_length, key_length)

        # Opitonally mask heads
        if layer_head_mask is not None:
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attention_output = tf.matmul(
            weights, value_states
        )  # (batch_size, num_heads, query_length, dim_per_head)

        attention_output = self.output_projector(unshape(attention_output))
        return (attention_output, position_bias)
