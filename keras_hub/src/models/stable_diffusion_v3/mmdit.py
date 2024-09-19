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
import math

import keras
from keras import layers
from keras import models
from keras import ops

from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.stable_diffusion_v3.mmdit_block import MMDiTBlock
from keras_hub.src.utils.keras_utils import standardize_data_format


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


class AdjustablePositionEmbedding(PositionEmbedding):
    def __init__(
        self,
        height,
        width,
        initializer="glorot_uniform",
        **kwargs,
    ):
        height = int(height)
        width = int(width)
        sequence_length = height * width
        super().__init__(sequence_length, initializer, **kwargs)
        self.height = height
        self.width = width

    def call(self, inputs, height=None, width=None):
        height = height or self.height
        width = width or self.width
        shape = ops.shape(inputs)
        feature_length = shape[-1]
        top = ops.floor_divide(self.height - height, 2)
        left = ops.floor_divide(self.width - width, 2)
        position_embedding = ops.convert_to_tensor(self.position_embeddings)
        position_embedding = ops.reshape(
            position_embedding, (self.height, self.width, feature_length)
        )
        position_embedding = ops.slice(
            position_embedding,
            (top, left, 0),
            (height, width, feature_length),
        )
        position_embedding = ops.reshape(
            position_embedding, (height * width, feature_length)
        )
        position_embedding = ops.expand_dims(position_embedding, axis=0)
        return position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape


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
                    embedding_dim, activation="silu", dtype=self.dtype_policy
                ),
                layers.Dense(
                    embedding_dim, activation=None, dtype=self.dtype_policy
                ),
            ],
            name="mlp",
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
                    num_modulation * hidden_dim, dtype=self.dtype_policy
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


class Unpatch(layers.Layer):
    def __init__(self, patch_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.output_dim = int(output_dim)

    def call(self, inputs, height, width):
        patch_size = self.patch_size
        output_dim = self.output_dim
        x = ops.reshape(
            inputs,
            (-1, height, width, patch_size, patch_size, output_dim),
        )
        # (b, h, w, p1, p2, o) -> (b, h, p1, w, p2, o)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(
            x,
            (-1, height * patch_size, width * patch_size, output_dim),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "output_dim": self.output_dim,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        return [inputs_shape[0], None, None, self.output_dim]


class MMDiT(keras.Model):
    def __init__(
        self,
        patch_size,
        num_heads,
        hidden_dim,
        depth,
        position_size,
        output_dim,
        mlp_ratio=4.0,
        latent_shape=(64, 64, 16),
        context_shape=(1024, 4096),
        pooled_projection_shape=(2048,),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if None in latent_shape:
            raise ValueError(
                "`latent_shape` must be fully specified. "
                f"Received: latent_shape={latent_shape}"
            )
        image_height = latent_shape[0] // patch_size
        image_width = latent_shape[1] // patch_size
        output_dim_in_final = patch_size**2 * output_dim
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError(
                "Currently only 'channels_last' is supported."
            )

        # === Layers ===
        self.patch_embedding = PatchEmbedding(
            patch_size,
            hidden_dim,
            data_format=data_format,
            dtype=dtype,
            name="patch_embedding",
        )
        self.position_embedding_add = layers.Add(
            dtype=dtype, name="position_embedding_add"
        )
        self.position_embedding = AdjustablePositionEmbedding(
            position_size, position_size, dtype=dtype, name="position_embedding"
        )
        self.context_embedding = layers.Dense(
            hidden_dim,
            dtype=dtype,
            name="context_embedding",
        )
        self.vector_embedding = models.Sequential(
            [
                layers.Dense(hidden_dim, activation="silu", dtype=dtype),
                layers.Dense(hidden_dim, activation=None, dtype=dtype),
            ],
            name="vector_embedding",
        )
        self.vector_embedding_add = layers.Add(
            dtype=dtype, name="vector_embedding_add"
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
                name=f"joint_block_{i}",
            )
            for i in range(depth)
        ]
        self.output_layer = OutputLayer(
            hidden_dim, output_dim_in_final, dtype=dtype, name="output_layer"
        )
        self.unpatch = Unpatch(
            patch_size, output_dim, dtype=dtype, name="unpatch"
        )

        # === Functional Model ===
        latent_inputs = layers.Input(shape=latent_shape, name="latent")
        context_inputs = layers.Input(shape=context_shape, name="context")
        pooled_projection_inputs = layers.Input(
            shape=pooled_projection_shape, name="pooled_projection"
        )
        timestep_inputs = layers.Input(shape=(1,), name="timestep")

        # Embeddings.
        x = self.patch_embedding(latent_inputs)
        position_embedding = self.position_embedding(
            x, height=image_height, width=image_width
        )
        x = self.position_embedding_add([x, position_embedding])
        context = self.context_embedding(context_inputs)
        pooled_projection = self.vector_embedding(pooled_projection_inputs)
        timestep_embedding = self.timestep_embedding(timestep_inputs)
        timestep_embedding = self.vector_embedding_add(
            [timestep_embedding, pooled_projection]
        )

        # Blocks.
        for block in self.joint_blocks:
            if block.use_context_projection:
                x, context = block(x, context, timestep_embedding)
            else:
                x = block(x, context, timestep_embedding)

        # Output layer.
        x = self.output_layer(x, timestep_embedding)
        outputs = self.unpatch(x, height=image_height, width=image_width)

        super().__init__(
            inputs={
                "latent": latent_inputs,
                "context": context_inputs,
                "pooled_projection": pooled_projection_inputs,
                "timestep": timestep_inputs,
            },
            outputs=outputs,
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
