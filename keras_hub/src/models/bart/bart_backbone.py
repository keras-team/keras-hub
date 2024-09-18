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
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


def bart_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.BartBackbone")
class BartBackbone(Backbone):
    """BART encoder-decoder network.

    This class implements a Transformer-based encoder-decoder model as
    described in
    ["BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"](https://arxiv.org/abs/1910.13461).

    The default constructor gives a fully customizable, randomly initialized BART
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer encoder layers and
            transformer decoder layers.
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
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "encoder_token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "encoder_padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        ),
        "decoder_token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "decoder_padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
        ),
    }

    # Pretrained BART encoder.
    model = keras_hub.models.BartBackbone.from_preset("bart_base_en")
    model(input_data)

    # Randomly initialized BART encoder-decoder model with a custom config
    model = keras_hub.models.BartBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
    )
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
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=bart_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.encoder_position_embedding = PositionEmbedding(
            initializer=bart_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="encoder_position_embedding",
        )
        self.encoder_embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="encoder_embeddings_add",
        )
        self.encoder_embeddings_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="encoder_embeddings_layer_norm",
        )
        self.encoder_embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_embeddings_dropout",
        )
        self.encoder_transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=keras.activations.gelu,
                dropout=dropout,
                layer_norm_epsilon=1e-5,
                kernel_initializer=bart_kernel_initializer(),
                dtype=dtype,
                name=f"transformer_encoder_layer_{i}",
            )
            self.encoder_transformer_layers.append(layer)
        self.decoder_position_embedding = PositionEmbedding(
            initializer=bart_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="decoder_position_embedding",
        )
        self.decoder_embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="decoder_embeddings_add",
        )
        self.decoder_embeddings_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="decoder_embeddings_layer_norm",
        )
        self.decoder_embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="decoder_embeddings_dropout",
        )
        self.decoder_transformer_layers = []
        for i in range(num_layers):
            layer = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=keras.activations.gelu,
                layer_norm_epsilon=1e-5,
                kernel_initializer=bart_kernel_initializer(),
                dtype=dtype,
                name=f"transformer_decoder_layer_{i}",
            )
            self.decoder_transformer_layers.append(layer)

        # === Functional Model ===
        encoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )
        # Encoder.
        tokens = self.token_embedding(encoder_token_id_input)
        positions = self.encoder_position_embedding(tokens)
        x = self.encoder_embeddings_add((tokens, positions))
        x = self.encoder_embeddings_layer_norm(x)
        x = self.encoder_embeddings_dropout(x)
        for transformer_layer in self.encoder_transformer_layers:
            x = transformer_layer(x, padding_mask=encoder_padding_mask_input)
        encoder_output = x
        # Decoder.
        tokens = self.token_embedding(decoder_token_id_input)
        positions = self.decoder_position_embedding(tokens)
        x = self.decoder_embeddings_add((tokens, positions))
        x = self.decoder_embeddings_layer_norm(x)
        x = self.decoder_embeddings_dropout(x)
        for transformer_layer in self.decoder_transformer_layers:
            x = transformer_layer(
                decoder_sequence=x,
                encoder_sequence=encoder_output,
                decoder_padding_mask=decoder_padding_mask_input,
                encoder_padding_mask=encoder_padding_mask_input,
            )
        decoder_output = x
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask_input,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask_input,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
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
