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

import keras
from keras import ops


class RelativeEmbedding(keras.layers.Layer):
    """Relative embedding layer.

    This is an implementation of relative embedding as described in the
    paper ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    This layer initializes an embedding matrix (of shape
    `(2 * batch_size, hidden_dim)`) for relative position encoding. It then
    applies layer normalization on the embedding matrix and returns the relative
    embedding matrix.

    Args:
        hidden_dim: int. The size of the dense embedding.
        bucket_size: int. The size of the relative position buckets.
        layer_norm_epsilon: float. Epsilon value to initialize the layer
            normalization layer.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense embedding.
            Defaults to `"glorot_uniform"`.
    """

    def __init__(
        self,
        hidden_dim,
        bucket_size,
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.bucket_size = bucket_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.rel_embeddings = self.add_weight(
            shape=(self.bucket_size * 2, self.hidden_dim),
            initializer=self.kernel_initializer,
            name="rel_embedding",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="rel_embeddings_layer_norm",
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]

        rel_embeddings = ops.expand_dims(
            ops.convert_to_tensor(self.rel_embeddings), axis=0
        )
        rel_embeddings = self.layer_norm(rel_embeddings)

        # Repeat `rel_embeddings` along axis = 0 `batch_size` times. The
        # resultant shape is `(batch_size, bucket_size * 2, hidden_dim)`.
        rel_embeddings = ops.repeat(rel_embeddings, repeats=batch_size, axis=0)

        return rel_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "bucket_size": self.bucket_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (self.bucket_size * 2, self.hidden_dim)
