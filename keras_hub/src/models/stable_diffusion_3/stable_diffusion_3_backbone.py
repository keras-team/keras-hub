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

    def compute_output_shape(self, inputs_shape):
        return (inputs_shape[0], self.hidden_dim)


class ClassifierFreeGuidanceConcatenate(layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(
        self,
        latents,
        positive_contexts,
        negative_contexts,
        positive_pooled_projections,
        negative_pooled_projections,
        timestep,
    ):
        timestep = ops.broadcast_to(timestep, ops.shape(latents)[:1])
        latents = ops.concatenate([latents, latents], axis=self.axis)
        contexts = ops.concatenate(
            [positive_contexts, negative_contexts], axis=self.axis
        )
        pooled_projections = ops.concatenate(
            [positive_pooled_projections, negative_pooled_projections],
            axis=self.axis,
        )
        timesteps = ops.concatenate([timestep, timestep], axis=self.axis)
        return latents, contexts, pooled_projections, timesteps

    def get_config(self):
        return super().get_config()


class ClassifierFreeGuidance(layers.Layer):
    """Perform classifier free guidance.

    This layer expects the inputs to be a concatenation of positive and negative
    (or empty) noise. The computation applies the classifier-free guidance
    scale.

    Args:
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Call arguments:
        inputs: A concatenation of positive and negative (or empty) noises.
        guidance_scale: The scale factor for classifier-free guidance.

    Reference:
    - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, guidance_scale):
        positive_noise, negative_noise = ops.split(inputs, 2, axis=0)
        return ops.add(
            negative_noise,
            ops.multiply(
                guidance_scale, ops.subtract(positive_noise, negative_noise)
            ),
        )

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        if outputs_shape[0] is not None:
            outputs_shape[0] = outputs_shape[0] // 2
        return outputs_shape


class EulerStep(layers.Layer):
    """A layer predicts the sample with the timestep and the predicted noise.

    Args:
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Call arguments:
        latents: A current sample created by the diffusion process.
        noise_residual: The direct output from the diffusion model.
        sigma: The amount of noise added at the current timestep.
        sigma_next: The amount of noise added at the next timestep.

    References:
    - [Common Diffusion Noise Schedules and Sample Steps are Flawed](
    https://arxiv.org/abs/2305.08891).
    - [Elucidating the Design Space of Diffusion-Based Generative Models](
    https://arxiv.org/abs/2206.00364).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, latents, noise_residual, sigma, sigma_next):
        sigma_diff = ops.subtract(sigma_next, sigma)
        return ops.add(latents, ops.multiply(sigma_diff, noise_residual))

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, latents_shape):
        return latents_shape


class LatentSpaceDecoder(layers.Layer):
    """Decoder to transform the latent space back to the original image space.

    During decoding, the latents are transformed back to the original image
    space using the equation: `latents / scale + shift`.

    Args:
        scale: float. The scaling factor.
        shift: float. The shift factor.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Call arguments:
        latents: The latent tensor to be transformed.

    Reference:
    - [High-Resolution Image Synthesis with Latent Diffusion Models](
    https://arxiv.org/abs/2112.10752).
    """

    def __init__(self, scale, shift, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.shift = shift

    def call(self, latents):
        return ops.add(ops.divide(latents, self.scale), self.shift)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale": self.scale,
                "shift": self.shift,
            }
        )
        return config

    def compute_output_shape(self, latents_shape):
        return latents_shape


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
        context_shape = (None, 4096 if t5 is None else t5.hidden_dim)
        pooled_projection_shape = (clip_l.hidden_dim + clip_g.hidden_dim,)

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
            context_shape=context_shape,
            pooled_projection_shape=pooled_projection_shape,
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
        # Set `dtype="float32"` to ensure the high precision for the noise
        # residual.
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            dtype="float32",
            name="scheduler",
        )
        self.cfg_concat = ClassifierFreeGuidanceConcatenate(
            dtype="float32", name="classifier_free_guidance_concat"
        )
        self.cfg = ClassifierFreeGuidance(
            dtype="float32", name="classifier_free_guidance"
        )
        self.euler_step = EulerStep(dtype="float32", name="euler_step")
        self.latent_space_decoder = LatentSpaceDecoder(
            scale=self.decoder.scaling_factor,
            shift=self.decoder.shift_factor,
            dtype="float32",
            name="latent_space_decoder",
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
            shape=(),
            dtype="int32",
            name="num_steps",
        )
        guidance_scale_input = keras.Input(
            shape=(),
            dtype="float32",
            name="guidance_scale",
        )
        embeddings = self.encode_step(token_ids, negative_token_ids)
        # Use `steps=0` to define the functional model.
        latents = self.denoise_step(
            latent_input,
            embeddings,
            0,
            num_step_input[0],
            guidance_scale_input[0],
        )
        outputs = self.decode_step(latents)
        inputs = {
            "latents": latent_input,
            "clip_l_token_ids": clip_l_token_id_input,
            "clip_l_negative_token_ids": clip_l_negative_token_id_input,
            "clip_g_token_ids": clip_g_token_id_input,
            "clip_g_negative_token_ids": clip_g_negative_token_id_input,
            "num_steps": num_step_input,
            "guidance_scale": guidance_scale_input,
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
        self.height = height
        self.width = width

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
        return (
            positive_embeddings,
            negative_embeddings,
            positive_pooled_embeddings,
            negative_pooled_embeddings,
        )

    def denoise_step(
        self,
        latents,
        embeddings,
        steps,
        num_steps,
        guidance_scale,
    ):
        steps = ops.convert_to_tensor(steps)
        steps_next = ops.add(steps, 1)
        sigma, timestep = self.scheduler(steps, num_steps)
        sigma_next, _ = self.scheduler(steps_next, num_steps)

        # Concatenation for classifier-free guidance.
        concated_latents, contexts, pooled_projs, timesteps = self.cfg_concat(
            latents, *embeddings, timestep
        )

        # Diffusion.
        predicted_noise = self.diffuser(
            {
                "latent": concated_latents,
                "context": contexts,
                "pooled_projection": pooled_projs,
                "timestep": timesteps,
            },
            training=False,
        )

        # Classifier-free guidance.
        predicted_noise = self.cfg(predicted_noise, guidance_scale)

        # Euler step.
        return self.euler_step(latents, predicted_noise, sigma, sigma_next)

    def decode_step(self, latents):
        latents = self.latent_space_decoder(latents)
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
                "height": self.height,
                "width": self.width,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to text encoders if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["clip_l"]["config"]:
                config["clip_l"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["clip_g"]["config"]:
                config["clip_g"]["config"]["dtype"] = dtype_config
            if (
                config["t5"] is not None
                and "dtype" not in config["t5"]["config"]
            ):
                config["t5"]["config"]["dtype"] = dtype_config

        # We expect `clip_l`, `clip_g` and/or `t5` to be instantiated.
        config["clip_l"] = layers.deserialize(
            config["clip_l"], custom_objects=custom_objects
        )
        config["clip_g"] = layers.deserialize(
            config["clip_g"], custom_objects=custom_objects
        )
        if config["t5"] is not None:
            config["t5"] = layers.deserialize(
                config["t5"], custom_objects=custom_objects
            )
        return cls(**config)
