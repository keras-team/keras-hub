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

from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.t5.t5_layer_norm import T5LayerNorm
from keras_hub.src.models.t5.t5_transformer_layer import T5TransformerLayer


class T5XXLTextEncoder(keras.Model):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        key_value_dim=None,
        dropout=0.1,
        activation="relu",
        use_gated_activation=True,
        layer_norm_epsilon=1e-06,
        tie_embedding_weights=True,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_embedding_weights,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            dtype=dtype,
            name="token_embedding",
        )
        self.encoder_embedding_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_embedding_dropout",
        )
        self.encoder_transformer_layers = []
        for i in range(num_layers):
            layer = T5TransformerLayer(
                is_decoder=False,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                key_value_dim=key_value_dim or hidden_dim // num_heads,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_gated_activation=use_gated_activation,
                use_relative_attention_bias=bool(i == 0),
                dtype=dtype,
                name=f"transformer_encoder_layer_{i}",
            )
            self.encoder_transformer_layers.append(layer)
        self.encoder_layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="encoder_output_layer_norm",
        )
        self.encoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_output_dropout",
        )

        # === Functional Model ===
        encoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )
        # Encoder.
        x = self.token_embedding(encoder_token_id_input)
        x = self.encoder_embedding_dropout(x)
        encoder_attention_mask = encoder_padding_mask_input[:, None, :]
        position_bias = None
        for transformer_layer in self.encoder_transformer_layers:
            output = transformer_layer(
                x,
                attention_mask=encoder_attention_mask,
                position_bias=position_bias,
                use_causal_mask=False,
            )
            if isinstance(output, tuple):
                x, position_bias = output
        x = self.encoder_layer_norm(x)
        x = self.encoder_dropout(x)
        encoder_output = x

        super().__init__(
            {
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask_input,
            },
            outputs=encoder_output,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.key_value_dim = key_value_dim
        self.dropout = dropout
        self.use_gated_activation = use_gated_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_embedding_weights = tie_embedding_weights

        if dtype is not None:
            try:
                self.dtype_policy = keras.dtype_policies.get(dtype)
            # Before Keras 3.2, there is no `keras.dtype_policies.get`.
            except AttributeError:
                if isinstance(dtype, keras.DTypePolicy):
                    dtype = dtype.name
                self.dtype_policy = keras.DTypePolicy(dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "activation": keras.activations.serialize(self.activation),
                "key_value_dim": self.key_value_dim,
                "dropout": self.dropout,
                "use_gated_activation": self.use_gated_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "tie_embedding_weights": self.tie_embedding_weights,
            }
        )
        return config
