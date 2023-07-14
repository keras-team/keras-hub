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

"""GPT-NeoX backbone model."""

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.gpt_neo_x.gpt_neo_x_decoder import GPTNeoXDecoder


def _gpt_neo_x_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.GPTNeoXBackbone")
class GPTNeoXBackbone(Backbone):
    """GPT-NeoX core network with hyperparameters.

    This network implements a Transformer-based decoder network,
    Generative Pretrained Transformer-Neo-X (GPTNeoX), as described in
    ["GPT-NeoX-20B: An Open-Source Autoregressive Language Model"](https://arxiv.org/abs/2204.06745).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    GPT-NeoX model with any number of layers, heads, and embedding
    dimensions.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/EleutherAI/gpt-neox/).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        layer_norm_epsilon: float. a value added to the denominator for
            numerical stability.
        rotary_max_wavelength: int. The maximum angular wavelength of the
            sine/cosine curves, for rotary embeddings.
        rotary_percentage: float. The percentage by which query, key, value
            matrices are to be rotated
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If `None`, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.0,
        rotary_percentage=0.25,
        rotary_max_wavelength=10000,
        layer_norm_epsilon=1e-5,
        max_sequence_length=512,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens
        token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_neo_x_kernel_initializer(stddev=0.01),
            name="token_embedding",
        )(token_ids)

        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(token_embedding)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = GPTNeoXDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_sequence_length=max_sequence_length,
                rotary_percentage=rotary_percentage,
                rotary_max_wavelength=rotary_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                kernel_initializer=_gpt_neo_x_kernel_initializer(stddev=0.02),
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=layer_norm_epsilon,
            dtype="float32",
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
        self.rotary_percentage = rotary_percentage
        self.rotary_max_wavelength = rotary_max_wavelength
        self.max_sequence_length = max_sequence_length
        self.layer_norm_epsilon = layer_norm_epsilon

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
                "rotary_percentage": self.rotary_percentage,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "max_sequence_length": self.max_sequence_length,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")
