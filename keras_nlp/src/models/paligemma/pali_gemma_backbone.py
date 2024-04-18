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

from keras_nlp.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.gemma.rms_normalization import RMSNormalization
from keras_nlp.src.models.paligemma.pali_gemma_decoder_block import (
    PaliGemmaDecoderBlock,
)


class PaliGemmaBackbone(Backbone):
    def __init__(
        self,
        img_sequence_length,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        self.img_sequence_length = img_sequence_length
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        #
        # Layers
        #
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
                seed=None,
            ),
            dtype=dtype,
            name="token_embedding",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            layer = PaliGemmaDecoderBlock(
                img_sequence_length=img_sequence_length,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                dropout=dropout,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        #
        # Functional Model
        #
        img_embeddings = keras.Input(
            shape=(img_sequence_length, hidden_dim),
            dtype=dtype,
            name="img_embeddings",
        )

        token_ids = keras.Input(
            shape=(None,),
            dtype="int32",
            name="token_ids",
        )

        padding_mask = keras.Input(
            shape=(None,), dtype="float32", name="padding_mask"
        )

        text_embeddings = self.token_embedding(token_ids)

        complete_sequence = keras.ops.concatenate(
            (img_embeddings, text_embeddings), axis=1
        )

        transformer_out = complete_sequence
        for transformer_layer in self.transformer_layers:
            transformer_out = transformer_layer(
                transformer_out, padding_mask=padding_mask
            )

        text_out = self.layer_norm(transformer_out)

        super().__init__(
            inputs={
                "img_embeddings": img_embeddings,
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=text_out,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_sequence_length": self.img_sequence_length,
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config
