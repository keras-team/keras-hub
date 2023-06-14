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
from keras_nlp.api_export import keras_nlp_export

@keras_nlp_export("keras_nlp.layers.MultiQueryAttentionBatched")
class MultiQueryAttentionBatched(keras.layers.Layer):
    """Multi-Query Attention Layer.

    This layer applies multi-query attention to the input tensor X using the key-value pairs from tensor M.
    The attention is computed by comparing the queries (Q) derived from X with the keys (K) derived from M.
    The resulting attention weights are applied to the values (V) derived from M to obtain the output tensor Y.

    Args:
        X: A tensor with shape [b, n, d].
        M: A tensor with shape [b, m, d].
        mask: A tensor with shape [b, h, n, m].
        P_q: A tensor with shape [h, d, k].
        P_k: A tensor with shape [d, k].
        P_v: A tensor with shape [d, v].
        P_o: A tensor with shape [h, d, v].

    Returns:
        Y: A tensor with shape [b, n, d].

    Examples:
        >>> X = tf.random.normal([32, 10, 64])
        >>> M = tf.random.normal([32, 20, 64])
        >>> mask = tf.random.normal([32, 8, 10, 20])
        >>> P_q = tf.random.normal([8, 64, 16])
        >>> P_k = tf.random.normal([64, 16])
        >>> P_v = tf.random.normal([64, 32])
        >>> P_o = tf.random.normal([8, 64, 32])
        >>> layer = MultiQueryAttentionBatched(P_q, P_k, P_v, P_o)
        >>> Y = layer([X, M, mask])
        >>> Y.shape
        TensorShape([32, 10, 64])

    References:
        - [Multi-Query Attention (Shazeer et al., 2019)](https://arxiv.org/abs/1911.02150)
    """
    def __init__(self, P_q, P_k, P_v, P_o, **kwargs):
        super(MultiQueryAttentionBatched, self).__init__(**kwargs)
        self.P_q = P_q
        self.P_k = P_k
        self.P_v = P_v
        self.P_o = P_o
    
    def call(self, inputs):
        X, M, mask = inputs
        Q = tf.einsum("bnd,hdk->bhnk", X, self.P_q)
        K = tf.einsum("bmd,dk->bmk", M, self.P_k)
        V = tf.einsum("bmd,dv->bmv", M, self.P_v)
        logits = tf.einsum("bhnk,bmk->bhnm", Q, K)
        weights = tf.softmax(logits + mask)
        O = tf.einsum("bhnm,bmv->bhnv", weights, V)
        Y = tf.einsum("bhnv,hdv->bnd", O, self.P_o)
        return Y

@keras_nlp_export("keras_nlp.layers.MultiQuerySelfAttentionIncremental")
class MultiQuerySelfAttentionIncremental(keras.layers.Layer):
    """Multi-Query Self-Attention Layer (One Step).

    This layer applies multi-query self-attention to the input tensor x using the previous key-value pairs.
    The attention is computed by comparing the query (q) derived from x with the keys (K) derived from the previous keys.
    The resulting attention weights are applied to the values (V) derived from the previous values to obtain the output tensor y.
    The updated key-value pairs are also returned.

    Args:
        x: A tensor with shape [b, d].
        prev_K: A tensor with shape [b, m, k].
        prev_V: A tensor with shape [b, m, v].
        P_q: A tensor with shape [h, d, k].
        P_k: A tensor with shape [d, k].
        P_v: A tensor with shape [d, v].
        P_o: A tensor with shape [h, d, v].

    Returns:
        y: A tensor with shape [b, d].
        new_K: A tensor with shape [b, m+1, k].
        new_V: A tensor with shape [b, m+1, v].

    Examples:
        >>> x = tf.random.normal([32, 64])
        >>> prev_K = tf.random.normal([32, 20, 16])
        >>> prev_V = tf.random.normal([32, 20, 32])
        >>> P_q = tf.random.normal([8, 64, 16])
        >>> P_k = tf.random.normal([64, 16])
        >>> P_v = tf.random.normal([64, 32])
        >>> P_o = tf.random.normal([8, 64, 32])
        >>> layer = MultiQuerySelfAttentionIncremental(P_q, P_k, P_v, P_o)
        >>> y, new_K, new_V = layer([x, prev_K, prev_V])
        >>> y.shape
        TensorShape([32, 64])
        >>> new_K.shape
        TensorShape([32, 21, 16])
        >>> new_V.shape
        TensorShape([32, 21, 32])

    References:
        - [Multi-Query Attention (Shazeer et al., 2019)](https://arxiv.org/abs/1911.02150)
    """

    def __init__(self, P_q, P_k, P_v, P_o, **kwargs):
        super(MultiQuerySelfAttentionIncremental, self).__init__(**kwargs)
        self.P_q = P_q
        self.P_k = P_k
        self.P_v = P_v
        self.P_o = P_o
    
    def call(self, inputs):
        x, prev_K, prev_V = inputs
        q = tf.einsum("bd,hdk->bhk", x, self.P_q)
        K = tf.concat([prev_K, tf.expand_dims(tf.einsum("bd,dk->bk", x, self.P_k), axis=2)], axis=2)
        V = tf.concat([prev_V, tf.expand_dims(tf.einsum("bd,dv->bv", x, self.P_v), axis=2)], axis=2)
        logits = tf.einsum("bhk,bmk->bhm", q, K)
        weights = tf.softmax(logits)
        O = tf.einsum("bhm,bmv->bhv", weights, V)
        y = tf.einsum("bhv,hdv->bd", O, self.P_o)
        return y, K, V
