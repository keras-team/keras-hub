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

from keras_hub.src.models.stable_diffusion_v3.vae_attention import VAEAttention
from keras_hub.src.utils.keras_utils import standardize_data_format


class VAEImageDecoder(keras.Model):
    def __init__(
        self,
        stackwise_num_filters,
        stackwise_num_blocks,
        output_channels=3,
        latent_shape=(None, None, 16),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        gn_axis = -1 if data_format == "channels_last" else 1

        # === Functional Model ===
        latent_inputs = layers.Input(shape=latent_shape)

        x = layers.Conv2D(
            stackwise_num_filters[0],
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name="input_projection",
        )(latent_inputs)
        x = apply_resnet_block(
            x,
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_block0",
        )
        x = VAEAttention(
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_attention",
        )(x)
        x = apply_resnet_block(
            x,
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_block1",
        )

        # Stacks.
        for i, filters in enumerate(stackwise_num_filters):
            for j in range(stackwise_num_blocks[i]):
                x = apply_resnet_block(
                    x,
                    filters,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"block{i}_{j}",
                )
            if i != len(stackwise_num_filters) - 1:
                # No upsamling in the last blcok.
                x = layers.UpSampling2D(
                    2,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"upsample_{i}",
                )(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    1,
                    padding="same",
                    data_format=data_format,
                    dtype=dtype,
                    name=f"upsample_{i}_conv",
                )(x)

        # Ouput block.
        x = layers.GroupNormalization(
            groups=32,
            axis=gn_axis,
            epsilon=1e-6,
            dtype=dtype,
            name="output_norm",
        )(x)
        x = layers.Activation("swish", dtype=dtype, name="output_activation")(x)
        image_outputs = layers.Conv2D(
            output_channels,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name="output_projection",
        )(x)
        super().__init__(inputs=latent_inputs, outputs=image_outputs, **kwargs)

        # === Config ===
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_num_blocks = stackwise_num_blocks
        self.output_channels = output_channels
        self.latent_shape = latent_shape

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
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "output_channels": self.output_channels,
                "image_shape": self.latent_shape,
            }
        )
        return config


def apply_resnet_block(x, filters, data_format=None, dtype=None, name=None):
    data_format = standardize_data_format(data_format)
    gn_axis = -1 if data_format == "channels_last" else 1
    input_filters = x.shape[gn_axis]

    residual = x
    x = layers.GroupNormalization(
        groups=32, axis=gn_axis, epsilon=1e-6, dtype=dtype, name=f"{name}_norm1"
    )(x)
    x = layers.Activation("swish", dtype=dtype)(x)
    x = layers.Conv2D(
        filters,
        3,
        1,
        padding="same",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_conv1",
    )(x)
    x = layers.GroupNormalization(
        groups=32, axis=gn_axis, epsilon=1e-6, dtype=dtype, name=f"{name}_norm2"
    )(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(
        filters,
        3,
        1,
        padding="same",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_conv2",
    )(x)
    if input_filters != filters:
        residual = layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_residual_projection",
        )(residual)
    x = layers.Add(dtype=dtype)([residual, x])
    return x
