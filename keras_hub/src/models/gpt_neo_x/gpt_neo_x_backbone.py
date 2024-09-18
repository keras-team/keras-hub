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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_decoder import GPTNeoXDecoder
from keras_hub.src.utils.keras_utils import gelu_approximate


def _gpt_neo_x_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.GPTNeoXBackbone")
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
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.
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
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_neo_x_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = GPTNeoXDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_sequence_length=max_sequence_length,
                rotary_percentage=rotary_percentage,
                rotary_max_wavelength=rotary_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=gelu_approximate,
                kernel_initializer=_gpt_neo_x_kernel_initializer(stddev=0.02),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="layer_norm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        # Embed tokens.
        x = self.token_embedding(token_id_input)
        x = self.embeddings_dropout(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
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
