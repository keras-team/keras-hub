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
from keras_nlp.models.bloom.bloom_decoder import BloomDecoder


def _bloom_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.BloomBackbone")
class BloomBackbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        max_sequence_length=512,
        **kwargs,
    ):
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens
        token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_bloom_kernel_initializer(stddev=0.02),
            tie_weights=False,
            name="token_embedding",
        )(token_ids)

        x = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="token_embedding_layernorm"
        )(token_embedding)

        for i in range(num_layers):
            x = BloomDecoder(
                num_heads=num_heads,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="final_layernorm"
        )(x)

        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.layer_norm_epsilon = layer_norm_epsilon
