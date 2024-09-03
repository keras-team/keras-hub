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
import math

import keras
from keras import layers
from keras import models
from keras import ops

from keras_nlp.src.models.stable_diffusion_v3.mmdit_block import MMDiTBlock
from keras_nlp.src.utils.keras_utils import standardize_data_format


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, hidden_dim, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        data_format = standardize_data_format(data_format)

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )

    def build(self, input_shape):
        self.patch_embedding.build(input_shape)

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        x_shape = ops.shape(x)
        x = ops.reshape(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class PositionEmbedding(layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, inputs):
        return ops.convert_to_tensor(self.position_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return list(self.position_embeddings.shape)


class TimestepEmbedding(layers.Layer):
    def __init__(
        self, embedding_dim, frequency_dim=256, max_period=10000, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        self.half_frequency_dim = self.frequency_dim // 2

        self.mlp = models.Sequential(
            [
                layers.Dense(
                    embedding_dim,
                    activation="silu",
                    dtype=self.dtype_policy,
                    name="dense0",
                ),
                layers.Dense(
                    embedding_dim,
                    activation=None,
                    dtype=self.dtype_policy,
                    name="dense1",
                ),
            ]
        )

    def build(self, inputs_shape):
        embedding_shape = list(inputs_shape)[1:]
        embedding_shape.append(self.frequency_dim)
        self.mlp.build(embedding_shape)

    def _create_timestep_embedding(self, inputs):
        compute_dtype = keras.backend.result_type(self.compute_dtype, "float32")
        x = ops.cast(inputs, compute_dtype)
        freqs = ops.exp(
            ops.divide(
                ops.multiply(
                    -math.log(self.max_period),
                    ops.arange(0, self.half_frequency_dim, dtype="float32"),
                ),
                self.half_frequency_dim,
            )
        )
        freqs = ops.cast(freqs, compute_dtype)
        x = ops.multiply(x, ops.expand_dims(freqs, axis=0))
        embedding = ops.concatenate([ops.cos(x), ops.sin(x)], axis=-1)
        if self.frequency_dim % 2 != 0:
            embedding = ops.pad(embedding, [[0, 0], [0, 1]])
        return ops.cast(embedding, self.compute_dtype)

    def call(self, inputs, training=None):
        timestep_embedding = self._create_timestep_embedding(inputs)
        return self.mlp(timestep_embedding, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_period": self.max_period,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        output_shape = list(inputs_shape)[1:]
        output_shape.append(self.embedding_dim)
        return output_shape


class OutputLayer(layers.Layer):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        num_modulation = 2

        self.adaptive_norm_modulation = models.Sequential(
            [
                layers.Activation("silu", dtype=self.dtype_policy),
                layers.Dense(
                    num_modulation * hidden_dim,
                    dtype=self.dtype_policy,
                    name="dense",
                ),
            ],
            name="adaptive_norm_modulation",
        )
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.output_dense = layers.Dense(
            output_dim,  # patch_size ** 2 * input_channels
            use_bias=True,
            dtype=self.dtype_policy,
            name="output_dense",
        )

    def build(self, inputs_shape, timestep_embedding_shape):
        self.adaptive_norm_modulation.build(timestep_embedding_shape)
        self.norm.build(inputs_shape)
        self.output_dense.build(inputs_shape)

    def _modulate(self, inputs, shift, scale):
        shift = ops.expand_dims(shift, axis=1)
        scale = ops.expand_dims(scale, axis=1)
        return ops.add(ops.multiply(inputs, ops.add(scale, 1.0)), shift)

    def call(self, inputs, timestep_embedding, training=None):
        x = inputs
        modulation = self.adaptive_norm_modulation(
            timestep_embedding, training=training
        )
        modulation = ops.reshape(modulation, (-1, 2, self.hidden_dim))
        shift, scale = ops.unstack(modulation, 2, axis=1)
        x = self._modulate(self.norm(x), shift, scale)
        x = self.output_dense(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            }
        )
        return config


class MMDiT(keras.Model):
    def __init__(
        self,
        patch_size,  # 2
        num_heads,  # 24
        hidden_dim,  # 64 * 24
        depth,  # 24
        position_size,  # 192
        output_dim,
        mlp_ratio=4.0,
        latent_shape=(64, 64, 16),
        context_shape=(1024, 4096),
        pooled_projection_shape=(2048,),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        position_sequence_length = position_size * position_size
        output_dim_in_final = patch_size**2 * output_dim

        # === Layers ===
        self.patch_embedding = PatchEmbedding(
            patch_size,
            hidden_dim,
            data_format=data_format,
            dtype=dtype,
            name="patch_embedding",
        )
        self.position_embedding = PositionEmbedding(
            position_sequence_length, dtype=dtype, name="position_embedding"
        )
        self.context_embedding = layers.Dense(
            hidden_dim,
            dtype=dtype,
            name="context_embedding",
        )
        self.vector_embedding = models.Sequential(
            [
                layers.Dense(
                    hidden_dim,
                    activation="silu",
                    dtype=dtype,
                    name="vector_embedding_dense_0",
                ),
                layers.Dense(
                    hidden_dim,
                    activation=None,
                    dtype=dtype,
                    name="vector_embedding_dense_1",
                ),
            ]
        )
        self.timestep_embedding = TimestepEmbedding(
            hidden_dim, dtype=dtype, name="timestep_embedding"
        )
        self.joint_blocks = [
            MMDiTBlock(
                num_heads,
                hidden_dim,
                mlp_ratio,
                use_context_projection=not (i == depth - 1),
                dtype=dtype,
                name=f"joint_block{i}",
            )
            for i in range(depth)
        ]
        self.final_layer = OutputLayer(
            hidden_dim, output_dim_in_final, dtype=dtype, name="final_layer"
        )

        # === Functional Model ===
        latent_inputs = layers.Input(shape=latent_shape, name="latent")
        context_inputs = layers.Input(shape=context_shape, name="context")
        pooled_projection_inputs = layers.Input(
            shape=pooled_projection_shape, name="pooled_projection"
        )
        timestep_inputs = layers.Input(shape=(1,), name="timestep")
        image_size = latent_shape[:2]

        # Embeddings.
        x = self.patch_embedding(latent_inputs)
        cropped_position_embedding = self._get_cropped_position_embedding(
            x, patch_size, image_size, position_size
        )
        x = layers.Add(dtype=dtype)([x, cropped_position_embedding])
        context = self.context_embedding(context_inputs)
        pooled_projection = self.vector_embedding(pooled_projection_inputs)
        timestep_embedding = self.timestep_embedding(timestep_inputs)
        timestep_embedding = layers.Add(dtype=dtype)(
            [timestep_embedding, pooled_projection]
        )

        # Blocks.
        for block in self.joint_blocks:
            if block.use_context_projection:
                x, context = block(x, context, timestep_embedding)
            else:
                x = block(x, context, timestep_embedding)

        # Final layer.
        x = self.final_layer(x, timestep_embedding)
        output_image = self._unpatchify(x, patch_size, image_size, output_dim)

        super().__init__(
            inputs={
                "latent": latent_inputs,
                "context": context_inputs,
                "pooled_projection": pooled_projection_inputs,
                "timestep": timestep_inputs,
            },
            outputs=output_image,
            **kwargs,
        )

        # === Config ===
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.position_size = position_size
        self.output_dim = output_dim
        self.mlp_ratio = mlp_ratio
        self.latent_shape = latent_shape
        self.context_shape = context_shape
        self.pooled_projection_shape = pooled_projection_shape

        if dtype is not None:
            try:
                self.dtype_policy = keras.dtype_policies.get(dtype)
            # Before Keras 3.2, there is no `keras.dtype_policies.get`.
            except AttributeError:
                if isinstance(dtype, keras.DTypePolicy):
                    dtype = dtype.name
                self.dtype_policy = keras.DTypePolicy(dtype)

    def _get_cropped_position_embedding(
        self, inputs, patch_size, image_size, position_size
    ):
        h, w = image_size
        h = h // patch_size
        w = w // patch_size
        top = (position_size - h) // 2
        left = (position_size - w) // 2
        hidden_dim = ops.shape(inputs)[-1]
        position_embedding = self.position_embedding(inputs)
        position_embedding = ops.reshape(
            position_embedding,
            (1, position_size, position_size, hidden_dim),
        )
        cropped_position_embedding = position_embedding[
            :, top : top + h, left : left + w, :
        ]
        cropped_position_embedding = ops.reshape(
            cropped_position_embedding, (1, h * w, hidden_dim)
        )
        return cropped_position_embedding

    def _unpatchify(self, x, patch_size, image_size, output_dim):
        h, w = image_size
        h = h // patch_size
        w = w // patch_size
        batch_size = ops.shape(x)[0]
        x = ops.reshape(
            x, (batch_size, h, w, patch_size, patch_size, output_dim)
        )
        # (b, h, w, p1, p2, o) -> (b, h, p1, w, p2, o)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(
            x, (batch_size, h * patch_size, w * patch_size, output_dim)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "depth": self.depth,
                "position_size": self.position_size,
                "output_dim": self.output_dim,
                "mlp_ratio": self.mlp_ratio,
                "latent_shape": self.latent_shape,
                "context_shape": self.context_shape,
                "pooled_projection_shape": self.pooled_projection_shape,
            }
        )
        return config
