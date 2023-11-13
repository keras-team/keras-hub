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
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.mistral.mistral_layer_norm import (
    MistralLayerNormalization,
)
from keras_nlp.models.mistral.mistral_transformer_decoder import (
    MistralTransformerDecoder,
)


def _mistral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.MistralBackbone")
class MistralBackbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        hidden_dim,
        intermediate_dim,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        sliding_window=512,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed Tokens
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_mistral_kernel_initializer(stddev=0.01),
            name="token_embedding",
        )
        x = token_embedding_layer(token_ids)

        # Apply successive transformer decoder blocks
        for i in range(num_layers):
            x = MistralTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_mistral_kernel_initializer(stddev=0.02),
                sliding_window=sliding_window,
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = MistralLayerNormalization(
            name="sequence_output_layernorm", epsilon=layer_norm_epsilon
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
        self.num_query_heads = num_query_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_heads = num_key_value_heads
        self.rope_scaling_factor = rope_scaling_factor
        self.sliding_window = sliding_window
        self.layer_norm_epsilon = layer_norm_epsilon
        self.token_embedding = token_embedding_layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "sliding_window": self.sliding_window,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
