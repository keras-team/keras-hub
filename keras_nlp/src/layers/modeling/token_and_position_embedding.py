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

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.modeling.position_embedding import PositionEmbedding
from keras_nlp.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_nlp.src.utils.keras_utils import clone_initializer


@keras_nlp_export("keras_nlp.layers.TokenAndPositionEmbedding")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """A layer which sums a token and position embedding.

    Token and position embeddings are ways of representing words and their order
    in a sentence. This layer creates a `keras.layers.Embedding` token embedding
    and a `keras_nlp.layers.PositionEmbedding` position embedding and sums their
    output when called. This layer assumes that the last dimension in the input
    corresponds to the sequence dimension.

    Args:
        vocabulary_size: The size of the vocabulary.
        sequence_length: The maximum length of input sequence
        embedding_dim: The output dimension of the embedding layer
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights.
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
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:
    ```python
    inputs = np.ones(shape=(1, 50), dtype="int32")
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=10_000,
        sequence_length=50,
        embedding_dim=128,
    )
    outputs = embedding_layer(inputs)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        embedding_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
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
        self.token_embedding = ReversibleEmbedding(
            vocabulary_size,
            embedding_dim,
            tie_weights=tie_weights,
            embeddings_initializer=clone_initializer(
                self.embeddings_initializer
            ),
            mask_zero=mask_zero,
            dtype=self.dtype_policy,
            name="token_embedding",
        )
        self.position_embedding = PositionEmbedding(
            sequence_length=sequence_length,
            initializer=clone_initializer(self.embeddings_initializer),
            dtype=self.dtype_policy,
            name="position_embedding",
        )
        self.supports_masking = self.token_embedding.supports_masking

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.token_embedding.build(input_shape)
        self.position_embedding.build(input_shape + (self.embedding_dim,))
        self.built = True

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
                "tie_weights": self.token_embedding.tie_weights,
                "mask_zero": self.token_embedding.mask_zero,
            }
        )
        return config

    def call(self, inputs, start_index=0):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(
            embedded_tokens,
            start_index=start_index,
        )
        outputs = embedded_tokens + embedded_positions
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.token_embedding.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.embedding_dim,)
