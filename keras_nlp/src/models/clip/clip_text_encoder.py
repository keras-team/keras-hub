# Copyright 2024 The KerasNLP Authors
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
from keras import layers

from keras_nlp.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.src.models.clip.clip_encoder_block import CLIPEncoderBlock


class CLIPTextEncoder(keras.Model):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        intermediate_output_index=None,
        max_sequence_length=77,
        dtype=None,
        **kwargs,
    ):
        if (
            intermediate_output_index is not None
            and intermediate_output_index < 0
        ):
            intermediate_output_index += num_layers

        # === Layers ===
        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            dtype=dtype,
            name="embedding",
        )
        self.encoder_layers = [
            CLIPEncoderBlock(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6, dtype="float32", name="layer_norm"
        )

        # === Functional Model ===
        token_id_input = layers.Input(
            shape=(max_sequence_length,), dtype="int32", name="token_ids"
        )
        x = self.embedding(token_id_input)
        intermediate_output = None
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i == intermediate_output_index:
                intermediate_output = x
        x = self.layer_norm(x)
        sequence_output = x

        if intermediate_output_index is not None:
            outputs = {
                "sequence_output": sequence_output,
                "intermediate_output": intermediate_output,
            }
        else:
            outputs = sequence_output
        super().__init__(
            inputs={"token_ids": token_id_input},
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.intermediate_output_index = intermediate_output_index

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
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "intermediate_output_index": self.intermediate_output_index,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
