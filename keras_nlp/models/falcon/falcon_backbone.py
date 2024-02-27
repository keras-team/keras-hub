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
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.falcon.falcon_transformer_decoder import (
    FalconTransformerDecoder,
)


@keras_nlp_export("keras_nlp.models.FalconBackbone")
class FalconBackbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_attention_heads,
        hidden_dim,
        intermediate_dim,
        layer_norm_epsilon=1e-5,
        attention_dropout=0,
        feedforward_dropout=0,
        **kwargs,
    ):
        # === Layers ===
        # Embed Tokens
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            name="token_embedding",
        )

        # Apply successive transformer decoder blocks.
        transformer_layers = []
        for i in range(num_layers):
            layer = FalconTransformerDecoder(
                num_attention_heads=num_attention_heads,
                intermediate_dim=intermediate_dim,
                attention_dropout=attention_dropout,
                feedforward_dropout=feedforward_dropout,
                name=f"transformer_layer_{i}",
            )
            transformer_layers.append(layer)

        layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="layer_norm",
        )

        # === Functional Model ===
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = token_embedding_layer(token_ids)

        for transformer_layer in transformer_layers:
            x = transformer_layer(inputs=x, decoder_padding_mask=padding_mask)
        sequence_output = layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
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
                "attention_dropout": self.attention_dropout,
                "feedforward_dropout": self.feedforward_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")
