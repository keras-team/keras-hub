import tensorflow as tf
from tensorflow import keras


# reference from https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/esm/modeling_tf_esm.py#L93

class RotaryEmbedding(keras.layers.Layer):

  def __init__(self, hidden_dim):
    super().__init__()
    self.hidden_dim = hidden_dim

  def build(self, input_shape):
    super().build(input_shape)
    self.inverse_freq = self.add_weight(
        "inverse_freq", shape=(self.hidden_dim // 2, ), dtype=tf.float32
    )
    self.inverse_freq.assign(
        1.0 / (10000 ** (tf.range(start=0, limit=self.hidden_dim, delta=2, dtype=tf.float32) / self.hidden_dim))
    )
  
  @staticmethod
  def apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
 
    cos_emb = cos_emb[:, :, :tf.shape(tensor)[-2], :]
    sin_emb = sin_emb[:, :, :tf.shape(tensor)[-2], :]
  
    x1, x2 = tf.split(tensor, 2, axis=-1)
    half_rot_tensor = tf.concat((-x2, x1), axis=-1)
 
    return (tensor*cos_emb) + (half_rot_tensor*sin_emb)

  def _compute_cos_sin(self, x, seq_dim=2):
    seq_len = tf.shape(x)[seq_dim]
    tensor = tf.range(seq_len, dtype=self.inverse_freq.dtype)
    freqs = tf.einsum("i, j -> ij", tensor, self.inverse_freq)
    embedding = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]
    return tf.cos(embedding), tf.sin(embedding)

  def call(self, query, key):
    cos_emb, sin_emb = self._compute_cos_sin(k, seq_dim=-2)
    return (
        self.apply_rotary_pos_emb(query, cos_emb, sin_emb),
        self.apply_rotary_pos_emb(key, cos_emb, sin_emb)
    )
