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
from keras_nlp.utils.keras_utils import clone_initializer


@keras.utils.register_keras_serializable(package="keras_nlp")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """A layer which sums a token and position embedding.

    This layer assumes that the last dimension in the input corresponds
    to the sequence dimension.

    Args:
        vocabulary_size: The size of the vocabulary.
        sequence_length: The maximum length of input sequence
        embedding_dim: The output dimension of the embedding layer
        embeddings_initializer: The initializer to use for the Embedding
            Layers
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which may take variable
            length input. If this is True, then all subsequent layers in the
            model need to support masking or an exception will be raised.
            If mask_zero` is set to True, as a consequence, index 0 cannot be
            used in the vocabulary
            (input_dim should equal size of vocabulary + 1).

    Examples:
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs = keras.Input(shape=(seq_length,))
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=seq_length,
        embedding_dim=embed_dim,
    )
    outputs = embedding_layer(inputs)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        embedding_dim,
        embeddings_initializer="glorot_uniform",
        mask_zero=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if vocabulary_size is None:
            raise ValueError(
                "`vocabulary_size` must be an Integer, received `None`."
            )
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        if embedding_dim is None:
            raise ValueError(
                "`embedding_dim` must be an Integer, received `None`."
            )
        self.vocabulary_size = int(vocabulary_size)
        self.sequence_length = int(sequence_length)
        self.embedding_dim = int(embedding_dim)
        self.embeddings_initializer = keras.initializers.get(
            embeddings_initializer
        )
        self.token_embedding = keras.layers.Embedding(
            vocabulary_size,
            embedding_dim,
            embeddings_initializer=clone_initializer(
                self.embeddings_initializer
            ),
            mask_zero=mask_zero,
            name="token_embedding"
            + str(keras.backend.get_uid("token_embedding")),
        )
        self.position_embedding = keras_nlp.layers.PositionEmbedding(
            sequence_length=sequence_length,
            initializer=clone_initializer(self.embeddings_initializer),
            name="position_embedding"
            + str(keras.backend.get_uid("position_embedding")),
        )
        self.supports_masking = self.token_embedding.supports_masking

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
                ),
                "mask_zero": self.token_embedding.mask_zero,
            },
        )
        return config

    def call(self, inputs):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(embedded_tokens)
        outputs = embedded_tokens + embedded_positions
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.token_embedding.compute_mask(inputs, mask=mask)
