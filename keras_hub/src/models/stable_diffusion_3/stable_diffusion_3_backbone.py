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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.stable_diffusion_3.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)
from keras_hub.src.models.stable_diffusion_3.mmdit import MMDiT
from keras_hub.src.models.stable_diffusion_3.vae_image_decoder import (
    VAEImageDecoder,
)
from keras_hub.src.utils.keras_utils import standardize_data_format


class CLIPProjection(layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)

        self.dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="dense",
        )

    def build(self, inputs_shape, token_ids_shape):
        inputs_shape = list(inputs_shape)
        self.dense.build([None, inputs_shape[-1]])

        # Assign identity matrix to the kernel as default.
        self.dense._kernel.assign(ops.eye(self.hidden_dim))

    def call(self, inputs, token_ids):
        indices = ops.expand_dims(
            ops.cast(ops.argmax(token_ids, axis=-1), "int32"), axis=-1
        )
        pooled_output = ops.take_along_axis(inputs, indices[:, :, None], axis=1)
        pooled_output = ops.squeeze(pooled_output, axis=1)
        return self.dense(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras_hub_export("keras_hub.models.StableDiffusion3Backbone")
class StableDiffusion3Backbone(Backbone):
    """Stable Diffusion 3 core network with hyperparameters.

    This backbone imports CLIP and T5 models as text encoders and implements the
    base MMDiT and VAE networks for the Stable Diffusion 3 model.

    The default constructor gives a fully customizable, randomly initialized
    MMDiT and VAE models with any hyperparameters. To load preset architectures
    and weights, use the `from_preset` constructor.

    Args:
        mmdit_patch_size: int. The size of each square patch in the input image
            in MMDiT.
        mmdit_hidden_dim: int. The size of the transformer hidden state at the
            end of each transformer layer in MMDiT.
        mmdit_num_layers: int. The number of transformer layers in MMDiT.
        mmdit_num_heads: int. The number of attention heads for each
            transformer in MMDiT.
        mmdit_position_size: int. The size of the height and width for the
            position embedding in MMDiT.
        vae_stackwise_num_filters: list of ints. The number of filters for each
            stack in VAE.
        vae_stackwise_num_blocks: list of ints. The number of blocks for each
            stack in VAE.
        clip_l: `keras_hub.models.CLIPTextEncoder`. The text encoder for
            encoding the inputs.
        clip_g: `keras_hub.models.CLIPTextEncoder`. The text encoder for
            encoding the inputs.
        t5: optional `keras_hub.models.T5Encoder`. The text encoder for
            encoding the inputs.
        latent_channels: int. The number of channels in the latent. Defaults to
            `16`.
        output_channels: int. The number of channels in the output. Defaults to
            `3`.
        num_train_timesteps: int. The number of diffusion steps to train the
            model. Defaults to `1000`.
        shift: float. The shift value for the timestep schedule. Defaults to
            `1.0`.
        height: optional int. The output height of the image.
        width: optional int. The output width of the image.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    # Pretrained Stable Diffusion 3 model.
    model = keras_hub.models.StableDiffusion3Backbone.from_preset(
        "stable_diffusion_3_medium"
    )

    # Randomly initialized Stable Diffusion 3 model with custom config.
    clip_l = keras_hub.models.CLIPTextEncoder(...)
    clip_g = keras_hub.models.CLIPTextEncoder(...)
    model = keras_hub.models.StableDiffusion3Backbone(
        mmdit_patch_size=2,
        mmdit_num_heads=4,
        mmdit_hidden_dim=256,
        mmdit_depth=4,
        mmdit_position_size=192,
        vae_stackwise_num_filters=[128, 128, 64, 32],
        vae_stackwise_num_blocks=[1, 1, 1, 1],
        clip_l=clip_l,
        clip_g=clip_g,
    )
    ```
    """

    def __init__(
        self,
        mmdit_patch_size,
        mmdit_hidden_dim,
        mmdit_num_layers,
        mmdit_num_heads,
        mmdit_position_size,
        vae_stackwise_num_filters,
        vae_stackwise_num_blocks,
        clip_l,
        clip_g,
        t5=None,
        latent_channels=16,
        output_channels=3,
        num_train_timesteps=1000,
        shift=1.0,
        height=None,
        width=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        height = int(height or 1024)
        width = int(width or 1024)
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "`height` and `width` must be divisible by 8. "
                f"Received: height={height}, width={width}"
            )
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        latent_shape = (height // 8, width // 8, latent_channels)

        # === Layers ===
        self.clip_l = clip_l
        self.clip_l_projection = CLIPProjection(
            clip_l.hidden_dim, dtype=dtype, name="clip_l_projection"
        )
        self.clip_l_projection.build([None, clip_l.hidden_dim], None)
        self.clip_g = clip_g
        self.clip_g_projection = CLIPProjection(
            clip_g.hidden_dim, dtype=dtype, name="clip_g_projection"
        )
        self.clip_g_projection.build([None, clip_g.hidden_dim], None)
        self.t5 = t5
        self.diffuser = MMDiT(
            mmdit_patch_size,
            mmdit_hidden_dim,
            mmdit_num_layers,
            mmdit_num_heads,
            mmdit_position_size,
            latent_shape=latent_shape,
            data_format=data_format,
            dtype=dtype,
            name="diffuser",
        )
        self.decoder = VAEImageDecoder(
            vae_stackwise_num_filters,
            vae_stackwise_num_blocks,
            output_channels,
            latent_shape=latent_shape,
            data_format=data_format,
            dtype=dtype,
            name="decoder",
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

        # === Functional Model ===
        latent_input = keras.Input(
            shape=latent_shape,
            name="latents",
        )
        clip_l_token_id_input = keras.Input(
            shape=(None,),
            dtype="int32",
            name="clip_l_token_ids",
        )
        clip_l_negative_token_id_input = keras.Input(
            shape=(None,),
            dtype="int32",
            name="clip_l_negative_token_ids",
        )
        clip_g_token_id_input = keras.Input(
            shape=(None,),
            dtype="int32",
            name="clip_g_token_ids",
        )
        clip_g_negative_token_id_input = keras.Input(
            shape=(None,),
            dtype="int32",
            name="clip_g_negative_token_ids",
        )
        token_ids = {
            "clip_l": clip_l_token_id_input,
            "clip_g": clip_g_token_id_input,
        }
        negative_token_ids = {
            "clip_l": clip_l_negative_token_id_input,
            "clip_g": clip_g_negative_token_id_input,
        }
        if self.t5 is not None:
            t5_token_id_input = keras.Input(
                shape=(None,),
                dtype="int32",
                name="t5_token_ids",
            )
            t5_negative_token_id_input = keras.Input(
                shape=(None,),
                dtype="int32",
                name="t5_negative_token_ids",
            )
            token_ids["t5"] = t5_token_id_input
            negative_token_ids["t5"] = t5_negative_token_id_input
        num_step_input = keras.Input(
            batch_shape=(),
            dtype="int32",
            name="num_steps",
        )
        cfg_scale_input = keras.Input(
            batch_shape=(),
            dtype="float32",
            name="classifier_free_guidance_scale",
        )
        embeddings = self.encode_step(token_ids, negative_token_ids)
        # Use `steps=0` to define the functional model.
        latents = self.denoise_step(
            latent_input, embeddings, 0, num_step_input, cfg_scale_input
        )
        outputs = self.decode_step(latents)
        inputs = {
            "latents": latent_input,
            "clip_l_token_ids": clip_l_token_id_input,
            "clip_l_negative_token_ids": clip_l_negative_token_id_input,
            "clip_g_token_ids": clip_g_token_id_input,
            "clip_g_negative_token_ids": clip_g_negative_token_id_input,
            "num_steps": num_step_input,
            "classifier_free_guidance_scale": cfg_scale_input,
        }
        if self.t5 is not None:
            inputs["t5_token_ids"] = t5_token_id_input
            inputs["t5_negative_token_ids"] = t5_negative_token_id_input
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.mmdit_patch_size = mmdit_patch_size
        self.mmdit_hidden_dim = mmdit_hidden_dim
        self.mmdit_num_layers = mmdit_num_layers
        self.mmdit_num_heads = mmdit_num_heads
        self.mmdit_position_size = mmdit_position_size
        self.vae_stackwise_num_filters = vae_stackwise_num_filters
        self.vae_stackwise_num_blocks = vae_stackwise_num_blocks
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        # We don't add `height` and `width` to config to make the backbone more
        # flexible.

    @property
    def latent_shape(self):
        return (None,) + tuple(self.diffuser.latent_shape)

    @property
    def clip_hidden_dim(self):
        return self.clip_l.hidden_dim + self.clip_g.hidden_dim

    @property
    def t5_hidden_dim(self):
        return 4096 if self.t5 is None else self.t5.hidden_dim

    def encode_step(self, token_ids, negative_token_ids):
        clip_hidden_dim = self.clip_hidden_dim
        t5_hidden_dim = self.t5_hidden_dim

        def encode(token_ids):
            clip_l_outputs = self.clip_l(token_ids["clip_l"], training=False)
            clip_g_outputs = self.clip_g(token_ids["clip_g"], training=False)
            clip_l_projection = self.clip_l_projection(
                clip_l_outputs["sequence_output"],
                token_ids["clip_l"],
                training=False,
            )
            clip_g_projection = self.clip_g_projection(
                clip_g_outputs["sequence_output"],
                token_ids["clip_g"],
                training=False,
            )
            pooled_embeddings = ops.concatenate(
                [clip_l_projection, clip_g_projection],
                axis=-1,
            )
            embeddings = ops.concatenate(
                [
                    clip_l_outputs["intermediate_output"],
                    clip_g_outputs["intermediate_output"],
                ],
                axis=-1,
            )
            embeddings = ops.pad(
                embeddings,
                [[0, 0], [0, 0], [0, t5_hidden_dim - clip_hidden_dim]],
            )
            if self.t5 is not None:
                t5_outputs = self.t5(token_ids["t5"], training=False)
                embeddings = ops.concatenate([embeddings, t5_outputs], axis=-2)
            else:
                padded_size = self.clip_l.max_sequence_length
                embeddings = ops.pad(
                    embeddings, [[0, 0], [0, padded_size], [0, 0]]
                )
            return embeddings, pooled_embeddings

        positive_embeddings, positive_pooled_embeddings = encode(token_ids)
        negative_embeddings, negative_pooled_embeddings = encode(
            negative_token_ids
        )

        # Concatenation for classifier-free guidance.
        embeddings = ops.concatenate(
            [positive_embeddings, negative_embeddings], axis=0
        )
        pooled_embeddings = ops.concatenate(
            [positive_pooled_embeddings, negative_pooled_embeddings], axis=0
        )
        return embeddings, pooled_embeddings

    def denoise_step(
        self,
        latents,
        embeddings,
        steps,
        num_steps,
        classifier_free_guidance_scale,
    ):
        contexts, pooled_projections = embeddings
        sigma = self.scheduler.get_sigma(steps, num_steps)
        sigma_next = self.scheduler.get_sigma(steps + 1, num_steps)

        # Sigma to timestep.
        timestep = self.scheduler._sigma_to_timestep(sigma)
        timestep = ops.broadcast_to(timestep, ops.shape(latents)[:1])

        # Diffusion.
        predicted_noise = self.diffuser(
            {
                "latent": ops.concatenate([latents, latents], axis=0),
                "context": contexts,
                "pooled_projection": pooled_projections,
                "timestep": ops.concatenate([timestep, timestep], axis=0),
            },
            training=False,
        )
        predicted_noise = ops.cast(predicted_noise, "float32")

        # Classifier-free guidance.
        classifier_free_guidance_scale = ops.cast(
            classifier_free_guidance_scale, predicted_noise.dtype
        )
        positive_noise, negative_noise = ops.split(predicted_noise, 2, axis=0)
        predicted_noise = negative_noise + classifier_free_guidance_scale * (
            positive_noise - negative_noise
        )

        # Euler step.
        return self.scheduler.step(latents, predicted_noise, sigma, sigma_next)

    def decode_step(self, latents):
        # Latent calibration.
        latents = ops.add(
            ops.divide(latents, self.decoder.scaling_factor),
            self.decoder.shift_factor,
        )

        # Decoding.
        return self.decoder(latents, training=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mmdit_patch_size": self.mmdit_patch_size,
                "mmdit_hidden_dim": self.mmdit_hidden_dim,
                "mmdit_num_layers": self.mmdit_num_layers,
                "mmdit_num_heads": self.mmdit_num_heads,
                "mmdit_position_size": self.mmdit_position_size,
                "vae_stackwise_num_filters": self.vae_stackwise_num_filters,
                "vae_stackwise_num_blocks": self.vae_stackwise_num_blocks,
                "clip_l": layers.serialize(self.clip_l),
                "clip_g": layers.serialize(self.clip_g),
                "t5": layers.serialize(self.t5),
                "latent_channels": self.latent_channels,
                "output_channels": self.output_channels,
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.shift,
            }
        )
        return config
