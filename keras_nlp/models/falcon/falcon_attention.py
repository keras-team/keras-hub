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

"""Falcon Attention"""
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from keras_nlp.models.falcon.falcon_rotary_positional_embedding import FalconRotaryPositionalEmbedding

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.FalconAttention")
class FalconAttention(keras.layers.Layer):
    """FalconAttention layer used in Falcon models.

    This layer implements the attention mechanism used in Falcon models.
    It performs multi-head self-attention on the input hidden states.

    Args:
        config: Configuration object containing the hyperparameters.

    Input Shapes:
        - hidden_states: Tensor of shape `(batch_size, seq_length, hidden_size)`.
        - alibi: Tensor of shape `(batch_size, num_heads, 1, kv_length)` containing
          alibi values for each head. Set to None if not using alibi.
        - attention_mask: Tensor of shape `(batch_size, 1, 1, seq_length)` containing
          the attention mask. Set to None if not using attention mask.
        - layer_past: Tuple of tensors `(past_key, past_value)` containing the cached
          key and value states from previous layers. Set to None if not using cache.
        - head_mask: Tensor of shape `(num_heads,)` containing the head mask. Set to None
          if not using head mask.
        - use_cache: Boolean value indicating whether to use the cache.
        - output_attentions: Boolean value indicating whether to output attention scores.

    Output Shapes:
        - output_tensor: Tensor of shape `(batch_size, seq_length, hidden_size)`.
        - present: Tuple of tensors `(key_layer, value_layer)` containing the updated
          key and value states for caching.
        - attention_probs (optional): Tensor of shape `(batch_size, num_heads, seq_length, seq_length)`
          containing the attention scores. Only present if `output_attentions=True`.

    Examples:
        ```python
        config = AttentionConfig(hidden_size=768, n_head=12, head_dim=64, hidden_dropout=0.1,
                                 bias=True, multi_query=False, attention_dropout=0.1)
        attention = FalconAttention(config)
        output_tensor, present = attention(hidden_states, alibi, attention_mask, layer_past=None,
                                           head_mask=None, use_cache=False, output_attentions=False)
        ```
    """

    def __init__(self, config):
        super(FalconAttention, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = FalconRotaryPositionalEmbedding(config.head_dim) if config.rotary else lambda q, k: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Dense(
            3 * self.hidden_size if not config.multi_query else (self.hidden_size + 2 * self.head_dim),
            bias=config.bias,
        )
        self.multi_query = config.multi_query
        self.dense = Dense(self.hidden_size, bias=config.bias)
        self.attention_dropout = Dropout(config.attention_dropout)
        self.num_kv = config.n_head if not self.multi_query else 1

    def _split_heads(self, fused_qkv):
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`
        Args:
            fused_qkv (`tf.Tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = tf.reshape(fused_qkv, (batch_size, seq_length, self.num_heads, 3, self.head_dim))
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = tf.reshape(fused_qkv, (batch_size, seq_length, self.num_heads + 2, self.head_dim))
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    def _merge_heads(self, x):
        """
        Merge heads together over the last dimension
        Args:
            x: (`tf.Tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]
        Returns:
            tf.Tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First reshape to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = tf.reshape(x, (batch_size, self.num_heads, seq_length, self.head_dim))

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = tf.transpose(x, (0, 2, 1, 3))

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return tf.reshape(x, (batch_size, seq_length, self.num_heads * self.head_dim))

    def call(
        self,
        hidden_states,
        alibi,
        attention_mask,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = tf.transpose(query_layer, (1, 2, 0, 3))
        query_layer = tf.reshape(query_layer, (q_length, batch_size * self.num_heads, self.head_dim))
        key_layer = tf.transpose(key_layer, (1, 2, 0, 3))
        key_layer = tf.reshape(key_layer, (q_length, batch_size * self.num_kv, self.head_dim))
        value_layer = tf.transpose(value_layer, (1, 2, 0, 3))
        value_layer = tf.reshape(value_layer, (q_length, batch_size * self.num_kv, self.head_dim))

        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = tf.concat((past_key, key_layer), axis=1)
            value_layer = tf.concat((past_value, value_layer), axis=1)

        _, kv_length, _ = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        if alibi is None:
            query_layer_ = tf.reshape(query_layer, (batch_size, self.num_heads, -1, self.head_dim))
            key_layer_ = tf.reshape(key_layer, (batch_size, self.num_kv, -1, self.head_dim))
            value_layer_ = tf.reshape(value_layer, (batch_size, self.num_kv, -1, self.head_dim))

            attn_scores = tf.matmul(query_layer_, key_layer_, transpose_b=True)
            attn_scores = tf.nn.softmax(attn_scores / tf.sqrt(tf.cast(tf.shape(key_layer_)[-1], tf.float32)))

            attn_output = tf.matmul(attn_scores, value_layer_)

            x = tf.reshape(attn_output, (batch_size, self.num_heads, q_length, self.head_dim))
            x = tf.transpose(x, (0, 2, 1, 3))
            attn_output = tf.reshape(x, (batch_size, q_length, self.num_heads * self.head_dim))

            output_tensor = self.dense(attn_output)

            outputs = (output_tensor, present)
            assert not output_attentions  # not supported.
            return outputs
        else:
            attention_mask_float = tf.cast(attention_mask * 1.0, dtype=tf.bfloat16)
            matmul_result = tf.matmul(query_layer, key_layer, transpose_b=True)

            # change shape to [batch_size, num_heads, q_length, kv_length]
            attention_scores = tf.reshape(matmul_result, (batch_size, self.num_heads, q_length, kv_length))

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == tf.float16 or input_dtype == tf.bfloat16:
                attention_scores = tf.cast(attention_scores, dtype=tf.float32)
            # attn_weights = tf.masked_fill(attention_scores, attention_mask, tf.finfo(attention_scores.dtype).min)
            attention_probs = tf.nn.softmax(
                (attention_scores + tf.reshape(alibi, (batch_size, self.num_heads, 1, -1))) * self.inv_norm_factor + attention_mask_float,
                axis=-1,
                dtype=hidden_states.dtype,
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change shape [batch_size x num_heads, q_length, kv_length]
            attention_probs_reshaped = tf.reshape(attention_probs, (batch_size * self.num_heads, q_length, kv_length))

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = tf.matmul(attention_probs_reshaped, value_layer)

            # change shape [batch_size, num_heads, q_length, head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            outputs = (output_tensor, present)
            if output_attentions:
                outputs += (attention_probs,)

            return outputs
