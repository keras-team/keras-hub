import tensorflow as tf
from tensorflow import keras
from keras_nlp.layers.transformer_layer_utils import compute_causal_mask
from rotary_embedding import RotaryEmbedding

class GPTNeoXAttention(keras.layers.Layer):

  def __init__(self, 
               num_heads, 
               hidden_dim, 
               rotary_pct=0.25, 
               max_position_embeddings=2048):
    
    super().__init__()
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.head_dim = hidden_dim // num_heads
    self.rotary_dim = self.head_dim * rotary_pct
    self.max_position_embeddings = max_position_embeddings
    self.rotary_embedding = RotaryEmbedding(self.rotary_pct)
    self.qkv = keras.layers.Dense(3 * self.hidden_dim) 
    self.dense = keras.layers.Dense(self.hidden_dim)

  def _compute_attention(self, 
                         query, 
                         key, 
                         value, 
                         attention_mask=None,
                         head_mask=None):
    
    batch_size, _, query_len, _ = tf.shape(query)
    key_len = tf.shape(key)[-2]
    # causal_mask = self.bias[:, :, key_len - query_len : key_len, :key_len]
    causal_mask = compute_causal_mask(batch_size, key_len, key_len)
    
    query = tf.reshape(query, [batch_size * self.num_heads, query_len, self.head_dim])
    key = tf.reshape(key, [batch_size * self.num_heads, query_len, self.head_dim])
    attention_scores = tf.zeros(
        [batch_size * self.num_heads, query_len, self.head_dim],
        dtype=query.dtype
    )

    attention_scores = tf.linalg.matmul(
        attention_scores,
        query,
        tf.transpose(key, perm=[0, 2, 1]),
        beta=1.0,
        alpha=(tf.constant(1.0))
    )
    attention_scores = tf.reshape(attention_scores,
                                  [batch_size, self.num_heads, query_len, key_len])
    mask_value = tf.constant(float('-inf'), dtype=attention_scores.dtype)
    attention_scores = tf.where(causal_mask, attention_scores, mask_value)

    if attention_mask is not None:
      attention_scores += attention_mask

    attention_scores = tf.cast(tf.nn.softmax(attention_scores, axis=-1), dtype=value.dtype)
    
    if head_mask is not None:
      attention_scores *= head_mask

    attention_output = tf.matmul(attention_scores, value)
    return attention_output, attention_scores


  def call(self, 
           hidden_states, 
           attention_mask, 
           head_mask, 
           layer_past, 
           return_attention_scores):

    qkv = self.qkv(hidden_states)
    new_qkv_shape = tf.shape(hidden_states)[:-1] + [self.num_heads, self.head_dim]
    qkv = tf.reshape(qkv, new_qkv_shape)

    query = tf.transpose(qkv[..., :self.head_dim], (0, 2, 1, 3))
    key = tf.transpose(qkv[..., :self.head_dim: 2*self.head_dim], (0, 2, 1, 3))
    value = tf.transpose(qkv[..., self.head_dim:], (0, 2, 1, 3))

    query_rot, query_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
    key_rot, key_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]
    
    query, key = self.rotary_embedding(query_rot, key_rot)
    query = tf.concat((query, query_pass), dim=-1)
    key = tf.concat((key, key_pass), dim=-1)

    if layer_past is not None:
      past_key, past_value = layer_past
      key = tf.concat((past_key, key), axis=-2)
      value = tf.concat((past_value, value), axis=-2)
    
    attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask, head_mask)
    new_shape = tf.shape(attention_output)[:-2] + (self.num_heads * self.head_dim)
    attention_output = tf.reshape(tf.transpose(attention_output, (0, 2, 1, 3)), new_shape)
    attention_output = self.dense(attention_output)
    
    if return_attention_scores:
      return (attention_output, attention_scores)

    return attention_output
