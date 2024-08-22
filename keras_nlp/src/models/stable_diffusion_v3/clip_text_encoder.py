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
from keras import layers
from keras import ops

from keras_nlp.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.src.models.stable_diffusion_v3.clip_encoder_block import (
    CLIPEncoderBlock,
)


class CLIPTextEncoder(layers.Layer):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation

        # Pre-defined parameters for StableDiffusionV3.
        self.vocabulary_size = 49408
        self.sequence_length = 77

        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=self.vocabulary_size,
            sequence_length=self.sequence_length,
            embedding_dim=embedding_dim,
            dtype=self.dtype_policy,
            name="embedding",
        )
        self.encoder_layers = [
            CLIPEncoderBlock(
                self.hidden_dim,
                self.num_heads,
                self.intermediate_dim,
                self.intermediate_activation,
                dtype=self.dtype_policy,
            )
            for _ in range(self.num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=0.00001, dtype=self.dtype_policy, name="layer_norm"
        )
        self.text_projection = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="text_projection",
        )

    def build(self, input_shape):
        self.embedding.build(input_shape)
        input_shape = self.embedding.compute_output_shape(input_shape)
        for layer in self.encoder_layers:
            layer.build(input_shape)
        self.layer_norm.build([None, self.sequence_length, self.hidden_dim])
        self.text_projection.build([None, self.hidden_dim])

        # Assign values to `self.text_projection`
        self.text_projection._kernel.assign(ops.eye(self.hidden_dim))

    def call(self, inputs, intermediate_output=None, training=None):
        if intermediate_output < 0:
            intermediate_output = self.num_layers + intermediate_output

        # Get embedding.
        x = self.embedding(inputs)

        # Generate causal mask.
        causal_mask = ops.full(
            (ops.shape(x)[1], ops.shape(x)[1]), float("-inf")
        )
        causal_mask = ops.triu(causal_mask, k=1)
        causal_mask = ops.cast(causal_mask, x.dtype)

        # Run through encoder.
        intermediate = None
        for i, block in enumerate(self.encoder_layers):
            x = block(x, attention_mask=causal_mask, training=training)
            if i == intermediate_output:
                intermediate = x
        x = self.layer_norm(x)
        if intermediate is not None:
            intermediate = self.layer_norm(intermediate)

        # Compute final projection.
        indices = ops.expand_dims(
            ops.cast(ops.argmax(inputs, axis=-1), "int32"), axis=-1
        )
        pooled_output = ops.take_along_axis(x, indices[:, :, None], axis=1)
        pooled_output = ops.squeeze(pooled_output, axis=1)
        out = self.text_projection(pooled_output)

        return x, intermediate, out, pooled_output
