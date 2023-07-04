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
        num_heads: The number of attention heads.
        hidden_dim: The dimensionality of the hidden states.
        dropout: The dropout rate to apply to the attention scores.
        max_sequence_length: The maximum sequence length.

    Input Shapes:
        - hidden_states: 3D tensor with shape `(batch_size, sequence_length, hidden_dim)`.

    Output Shapes:
        - outputs: 3D tensor with shape `(batch_size, sequence_length, hidden_dim)`.

    Examples:
        ```python
        falcon_attention = FalconAttention(num_heads=8, hidden_dim=512, dropout=0.1)
        outputs = falcon_attention(hidden_states)
        ```
    """
    def __init__(self,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_sequence_length=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.split_size = self.hidden_dim
        self.dropout = dropout
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"`hidden_dim` must be divisible by num_heads (got `hidden_dim`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.maybe_rotary = FalconRotaryPositionalEmbedding(self.head_dim)
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        self.query_key_value = tf.keras.layers.Dense(3 * self.hidden_dim)
        
        self.multi_query = True 
        self.dense = tf.keras.layers.Dense(self.hidden_dim)
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.num_kv = self.num_heads 

    def build(self, input_shape):
        self.query_key_value = tf.keras.layers.Dense(3 * self.hidden_dim, use_bias=False)
        super(FalconAttention, self).build(input_shape)
    def _split_heads(self, fused_qkv):
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape.as_list()
        hidden_size = self.num_heads * self.head_dim
        fused_qkv = tf.reshape(fused_qkv, (batch_size, seq_length, self.num_heads, -1))
        return fused_qkv[..., :hidden_size], fused_qkv[..., hidden_size:2*hidden_size], fused_qkv[..., 2*hidden_size:]

    # def _split_heads(self, fused_qkv):
    #     batch_size, seq_length, hidden_size = fused_qkv.shape.as_list()
    #     hidden_size = self.num_heads * self.head_dim
    #     fused_qkv = tf.reshape(fused_qkv, (batch_size, seq_length, self.num_heads, -1))
    #     return fused_qkv[..., :self.head_dim], fused_qkv[..., self.head_dim:2*self.head_dim], fused_qkv[..., 2*self.head_dim:]

    def _merge_heads(self, x):
            batch_size_and_num_heads, seq_length, _ = x.shape.as_list()
            batch_size = batch_size_and_num_heads // self.num_heads
            x = tf.reshape(x, (batch_size, self.num_heads, seq_length, self.head_dim))
            x = tf.transpose(x, (0, 2, 1, 3))
            return tf.reshape(x, (batch_size, seq_length, self.num_heads * self.head_dim))
    
    def scaled_dot_product_attention(query, key, value, mask, dropout_rate, is_causal=True):
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.einsum('...qd,...kd->...qk', query, key) / tf.math.sqrt(d_k)
        
        if is_causal:
            length = tf.shape(scores)[-1]
            causal_mask = tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((length, length))
            ).to_dense()
            causal_mask = tf.reshape(causal_mask, [1, 1, length, length])
            scores = scores * causal_mask + (1.0 - causal_mask) * tf.float32.min
        
        if mask is not None:
            scores += mask * tf.float32.min
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_rate)
        
        output = tf.einsum('...qk,...kd->...qd', attention_weights, value)
        
        return output, attention_weights
    
    def call(self, hidden_states, attention_mask=None, alibi=None, layer_past=None, head_mask=None,
            use_cache=False, output_attentions=False, return_attention_scores=False, training=None):
        
        # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = self.query_key_value(hidden_states)  

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = tf.split(fused_qkv, 3, axis=-1)

        batch_size, q_length, _, _ = tf.shape(query_layer)

        query_layer = tf.transpose(query_layer, [0, 2, 1, 3])
        query_layer = tf.reshape(query_layer, [batch_size * self.num_heads, q_length, self.head_dim])
        key_layer = tf.transpose(key_layer, [0, 2, 1, 3])
        key_layer = tf.reshape(key_layer, [batch_size * self.num_kv, q_length, self.head_dim])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        value_layer = tf.reshape(value_layer, [batch_size * self.num_kv, q_length, self.head_dim])

        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = tf.concat([past_key, key_layer], axis=1)
            value_layer = tf.concat([past_value, value_layer], axis=1)

        _, kv_length, _ = tf.shape(key_layer)

        query_layer_ = tf.reshape(query_layer, [batch_size, self.num_heads, -1, self.head_dim])
        key_layer_ = tf.reshape(key_layer, [batch_size, self.num_kv, -1, self.head_dim])
        value_layer_ = tf.reshape(value_layer, [batch_size, self.num_kv, -1, self.head_dim])


        attn_output = self.scaled_dot_product_attention(
            query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True
        )

        x = tf.reshape(attn_output, [batch_size, self.num_heads, q_length, self.head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        attn_output = tf.reshape(x, [batch_size, q_length, self.num_heads * self.head_dim])

        output_tensor = self.dense(attn_output)

        outputs = (output_tensor)
        return outputs
