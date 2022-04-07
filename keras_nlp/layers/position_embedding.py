# Copyright 2022 The KerasNLP Authors
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

"""Position embedding implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

SEQUENCE_AXIS = -2


class PositionEmbedding(keras.layers.Layer):
    """Creates a layer which learns a position embedding for inputs sequences.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This class accepts `RaggedTensor`s as inputs to process batches of sequences
    of different lengths. The one ragged dimension must be the dimension that
    corresponds to the sequence, that is, the penultimate dimension.

    Args:
        max_length: The maximum length of the dynamic sequence.
        initializer: The initializer to use for the embedding weights. Defaults
            to "glorot_uniform".
        seq_axis: The axis of the input tensor where we add the embeddings.

    Example:
    ```python
    token_embeddings = layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        max_length=max_length
    )

    embedded_tokens = token_embeddings(inputs)
    embedded_positions = position_embeddings(embedded_tokens)
    outputs = embedded_tokens + embedded_positions
    ```

    Reference:
        [BERT: Pre-training of Deep Bidirectional Transformers for Language
        Understanding](https://arxiv.org/abs/1810.04805).
    """

    def __init__(
        self,
        max_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_length is None:
            raise ValueError("`max_length` must be an Integer, not `None`.")
        self.max_length = int(max_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_length": self.max_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            "embeddings",
            shape=[self.max_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            bounding_shape = inputs.bounding_shape()
            position_embeddings = self._trim_and_broadcast_position_embeddings(
                bounding_shape,
            )
            # then apply row lengths to recreate the same ragged shape as inputs
            return tf.RaggedTensor.from_tensor(
                position_embeddings,
                inputs.nested_row_lengths(),
            )
        else:
            return self._trim_and_broadcast_position_embeddings(
                tf.shape(inputs),
            )

    def _trim_and_broadcast_position_embeddings(self, shape):
        sequence_length = shape[SEQUENCE_AXIS]
        # trim to match the length of the sequence
        position_embeddings = self.position_embeddings[:sequence_length, :]
        # then broadcast to add the missing dimensions to match "shape"
        return tf.broadcast_to(position_embeddings, shape)
