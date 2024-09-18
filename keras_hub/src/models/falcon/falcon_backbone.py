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
from keras_hub.src.models.falcon.falcon_transformer_decoder import (
    FalconTransformerDecoder,
)


@keras_hub_export("keras_hub.models.FalconBackbone")
class FalconBackbone(Backbone):
    """The Falcon core architecure.

    This network implements a Transformer-based decoder-only network,
    [Falcon](https://arxiv.org/abs/2306.01116).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_attention_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The dimensionality of the embeddings and hidden states.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the MLP network of each transformer.
        layer_norm_epsilon: float. Epsilon for the layer normalization layers in
            the transformer decoder.
        attention_dropout_rate: float. Dropout probability for the attention.
        feedforward_dropout_rate: flaot. Dropout probability for the feedforward.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Falcon decoder.
    # TODO: Update the preset.
    model = keras_hub.models.FalconBackbone.from_preset("falcon_preset")
    model(input_data)

    # Randomly initialized Falcon decoder with a custom config.
    model = keras_hub.models.FalconBackbone(
        vocabulary_size=10,
        num_layers=2,
        num_attention_heads=2,
        hidden_dim=32,
        intermediate_dim=32*4,
        layer_norm_epsilon=1e-5,
        attention_dropout_rate=0,
        feedforward_dropout_rate=0,
        dtype="float32",
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_attention_heads,
        hidden_dim,
        intermediate_dim,
        layer_norm_epsilon=1e-5,
        attention_dropout_rate=0,
        feedforward_dropout_rate=0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="token_embedding",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            layer = FalconTransformerDecoder(
                num_attention_heads=num_attention_heads,
                intermediate_dim=intermediate_dim,
                attention_dropout_rate=attention_dropout_rate,
                feedforward_dropout_rate=feedforward_dropout_rate,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.final_layernorm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        # Embed Tokens.
        x = self.token_embedding(token_ids)

        # Apply successive transformer decoder blocks.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(inputs=x, decoder_padding_mask=padding_mask)
        sequence_output = self.final_layernorm(x)

        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.feedforward_dropout_rate = feedforward_dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "attention_dropout_rate": self.attention_dropout_rate,
                "feedforward_dropout_rate": self.feedforward_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
