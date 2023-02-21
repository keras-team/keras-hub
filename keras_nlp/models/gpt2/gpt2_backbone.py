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

"""GPT-2 backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import PositionEmbedding
from keras_nlp.layers import TransformerDecoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


def _gpt_2_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2Backbone(Backbone):
    """GPT-2 core network with hyperparameters.

    This network implements a Transformer-based decoder network,
    Generative Pretrained Transformer-2 (GPT-2), as described in
    ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    GPT-2 model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Example usage:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }

    # Pretrained GPT-2 decoder
    model = GPT2Backbone.from_preset("gpt2_base_en")
    output = model(input_data)

    # Randomly initialized GPT-2 decoder with custom config
    model = keras_nlp.models.GPT2Backbone(
        vocabulary_size=50257,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=1024,
    )

    # Call the model on the input data.
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=1024,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions.
        token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_2_kernel_initializer(stddev=0.01),
            name="token_embedding",
        )(token_ids)

        # Can't use `TokenAndPositionEmbedding` layer here because of different
        # initializers.
        position_embedding = PositionEmbedding(
            initializer=_gpt_2_kernel_initializer(stddev=0.02),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)

        # Sum and apply dropout to embeddings.
        x = keras.layers.Add()((token_embedding, position_embedding))
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm_epsilon=1e-05,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                normalize_first=True,
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=1e-05,
            dtype=tf.float32,
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
