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

"""Creates an Embedding Layer and adds Positional Embeddings"""

from tensorflow import keras

import keras_nlp.layers


class TokenAndPositionEmbedding(keras.layers.Layer):
    """A layer which sums a token and position embedding.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    Args:
        `vocabulary_size`: The size of the vocabulary (should be no larger than 999)
        `max_length`: The maximum length of input sequence
        `embedding_dim`: The output dimension of the embedding layer
        `embeddings_initializer`: The initializer to use for the Embedding Layer
        `position_embeddings_initializer`: The initializer to use for Position
            Embedding Layer
        `embeddings_regularizer`: The regularizer to user for regularization of the
            Embedding Layer
        `mask_zero`: Boolean, whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using recurrent layers which may take variable
            length input. If this is True, then all subsequent layers in the
            model need to support masking or an exception will be raised. If mask_zero
            is set to True, as a consequence, index 0 cannot be used in the vocabulary
            (input_dim should equal size of vocabulary + 1).
        `seq_axis`: The axis of the input tensor where we add the embeddings

    Example:
    ```python
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        max_length=max_length,
        embedding_dim=embed_dim,
    )
    outputs = embedding_layer(inputs)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        max_length,
        embedding_dim,
        embeddings_initializer="glorot_uniform",
        position_embeddings_initializer="glorot_uniform",
        embeddings_regularizer=None,
        mask_zero=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if vocabulary_size is None:
            raise ValueError(
                "`vocabulary_size` must be an Integer, received `None`."
            )
        if max_length is None:
            raise ValueError(
                "`max_length` must be an Integer, received `None`."
            )
        if embedding_dim is None:
            raise ValueError(
                "`embedding_dim` must be an Integer, received `None`."
            )
        self.vocabulary_size = int(vocabulary_size)
        self.max_length = int(max_length)
        self.embedding_dim = int(embedding_dim)
        self.token_embedding = keras.layers.Embedding(
            vocabulary_size,
            embedding_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            mask_zero=mask_zero,
        )
        self.position_embedding = keras_nlp.layers.PositionEmbedding(
            max_length=max_length,
            initializer=position_embeddings_initializer,
            **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
                "embeddings_initializer": keras.initializers.serialize(
                    self.token_embedding.embeddings_initializer
                ),
                "embeddings_regularizer": keras.regularizers.serialize(
                    self.token_embedding.embeddings_regularizer
                ),
                "position_embeddings_initializer": keras.initializers.serialize(
                    self.position_embedding.initializer
                ),
                "mask_zero": self.token_embedding.mask_zero,
            },
        )
        return config

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(embedded_tokens)
        outputs = embedded_tokens + embedded_positions
        return outputs
