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
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.stable_diffusion_v3.clip_encoder_block import (
    CLIPEncoderBlock,
)


class CLIPTextEncoder(keras.Model):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        intermediate_output_index=None,
        vocabulary_size=49408,
        sequence_length=77,
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
            sequence_length=sequence_length,
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
            epsilon=0.00001, dtype=dtype, name="layer_norm"
        )
        self.text_projection = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="text_projection",
        )

        # === Functional Model ===
        encoder_token_ids = layers.Input(
            shape=(sequence_length,), dtype="int32", name="encoder_token_ids"
        )
        x = self.embedding(encoder_token_ids)
        encoder_intermediate_output = None
        # Encoder.
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i == intermediate_output_index:
                encoder_intermediate_output = x
        x = self.layer_norm(x)
        encoder_output = x
        if encoder_intermediate_output is not None:
            encoder_intermediate_output = self.layer_norm(
                encoder_intermediate_output
            )
        # Projection.
        indices = ops.expand_dims(
            ops.cast(ops.argmax(encoder_token_ids, axis=-1), "int32"), axis=-1
        )
        pooled_output = ops.take_along_axis(x, indices[:, :, None], axis=1)
        pooled_output = ops.squeeze(pooled_output, axis=1)
        projection_output = self.text_projection(pooled_output)

        outputs = {
            "encoder_sequence_output": encoder_output,
            "encoder_pooled_output": pooled_output,
            "encoder_projection_output": projection_output,
        }
        if intermediate_output_index is not None:
            outputs["encoder_intermediate_output"] = encoder_intermediate_output

        super().__init__(
            inputs={"encoder_token_ids": encoder_token_ids},
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.intermediate_output_index = intermediate_output_index
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length

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
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "intermediate_output_index": self.intermediate_output_index,
                "vocabulary_size": self.vocabulary_size,
                "sequence_length": self.sequence_length,
            }
        )
        return config
