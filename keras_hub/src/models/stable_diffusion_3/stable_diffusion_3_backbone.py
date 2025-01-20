import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.stable_diffusion_3.flow_match_euler_discrete_scheduler import (  # noqa: E501
    FlowMatchEulerDiscreteScheduler,
)
from keras_hub.src.models.stable_diffusion_3.mmdit import MMDiT
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


class CLIPConcatenate(layers.Layer):
    def call(
        self,
        clip_l_projection,
        clip_g_projection,
        clip_l_intermediate_output,
        clip_g_intermediate_output,
        padding,
    ):
        pooled_embeddings = ops.concatenate(
            [clip_l_projection, clip_g_projection], axis=-1
        )
        embeddings = ops.concatenate(
            [clip_l_intermediate_output, clip_g_intermediate_output], axis=-1
        )
        embeddings = ops.pad(embeddings, [[0, 0], [0, 0], [0, padding]])
        return pooled_embeddings, embeddings


class ImageRescaling(layers.Rescaling):
    """Rescales inputs from image space to latent space.

    The rescaling is performed using the formula: `(inputs - offset) * scale`.
    """

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = self.backend.cast(self.scale, dtype)
        offset = self.backend.cast(self.offset, dtype)
        return (self.backend.cast(inputs, dtype) - offset) * scale


class LatentRescaling(layers.Rescaling):
    """Rescales inputs from latent space to image space.

    The rescaling is performed using the formula: `inputs / scale + offset`.
    """

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = self.backend.cast(self.scale, dtype)
        offset = self.backend.cast(self.offset, dtype)
        return (self.backend.cast(inputs, dtype) / scale) + offset


class ClassifierFreeGuidanceConcatenate(layers.Layer):
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
        latents = ops.concatenate([latents, latents], axis=0)
        contexts = ops.concatenate(
            [positive_contexts, negative_contexts], axis=0
        )
        pooled_projections = ops.concatenate(
            [positive_pooled_projections, negative_pooled_projections], axis=0
        )
        timesteps = ops.concatenate([timestep, timestep], axis=0)
        return latents, contexts, pooled_projections, timesteps


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

    def call(self, inputs, guidance_scale):
        positive_noise, negative_noise = ops.split(inputs, 2, axis=0)
        return ops.add(
            negative_noise,
            ops.multiply(
                guidance_scale, ops.subtract(positive_noise, negative_noise)
            ),
        )

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

    def call(self, latents, noise_residual, sigma, sigma_next):
        sigma_diff = ops.subtract(sigma_next, sigma)
        return ops.add(latents, ops.multiply(sigma_diff, noise_residual))

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
        mmdit_qk_norm: Optional str. Whether to normalize the query and key
            tensors for each transformer in MMDiT. Available options are `None`
            and `"rms_norm"`. Typically, this is set to `None` for 3.0 version
            and to `"rms_norm"` for 3.5 version.
        mmdit_dual_attention_indices: Optional tuple. Specifies the indices of
            the blocks that serve as dual attention blocks. Typically, this is
            for 3.5 version. Defaults to `None`.
        vae: The VAE used for transformations between pixel space and latent
            space.
        clip_l: The CLIP text encoder for encoding the inputs.
        clip_g: The CLIP text encoder for encoding the inputs.
        t5: optional The T5 text encoder for encoding the inputs.
        latent_channels: int. The number of channels in the latent. Defaults to
            `16`.
        output_channels: int. The number of channels in the output. Defaults to
            `3`.
        num_train_timesteps: int. The number of diffusion steps to train the
            model. Defaults to `1000`.
        shift: float. The shift value for the timestep schedule. Defaults to
            `3.0`.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(1024, 1024, 3)`.
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
    vae = keras_hub.models.VAEBackbone(...)
    clip_l = keras_hub.models.CLIPTextEncoder(...)
    clip_g = keras_hub.models.CLIPTextEncoder(...)
    model = keras_hub.models.StableDiffusion3Backbone(
        mmdit_patch_size=2,
        mmdit_num_heads=4,
        mmdit_hidden_dim=256,
        mmdit_depth=4,
        mmdit_position_size=192,
        mmdit_qk_norm=None,
        mmdit_dual_attention_indices=None,
        vae=vae,
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
        mmdit_qk_norm,
        mmdit_dual_attention_indices,
        vae,
        clip_l,
        clip_g,
        t5=None,
        latent_channels=16,
        output_channels=3,
        num_train_timesteps=1000,
        shift=3.0,
        image_shape=(1024, 1024, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        height = image_shape[0]
        width = image_shape[1]
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "height and width in `image_shape` must be divisible by 8. "
                f"Received: image_shape={image_shape}"
            )
        latent_shape = (height // 8, width // 8, int(latent_channels))
        context_shape = (None, 4096 if t5 is None else t5.hidden_dim)
        pooled_projection_shape = (clip_l.hidden_dim + clip_g.hidden_dim,)
        self._latent_shape = latent_shape

        # === Layers ===
        self.clip_l = clip_l
        self.clip_l_projection = CLIPProjection(
            clip_l.hidden_dim, dtype=dtype, name="clip_l_projection"
        )
        self.clip_g = clip_g
        self.clip_g_projection = CLIPProjection(
            clip_g.hidden_dim, dtype=dtype, name="clip_g_projection"
        )
        self.clip_concatenate = CLIPConcatenate(
            dtype=dtype, name="clip_concatenate"
        )
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
            qk_norm=mmdit_qk_norm,
            dual_attention_indices=mmdit_dual_attention_indices,
            data_format=data_format,
            dtype=dtype,
            name="diffuser",
        )
        self.vae = vae
        self.cfg_concat = ClassifierFreeGuidanceConcatenate(
            dtype=dtype, name="classifier_free_guidance_concat"
        )
        self.cfg = ClassifierFreeGuidance(
            dtype=dtype, name="classifier_free_guidance"
        )
        # Set `dtype="float32"` to ensure the high precision for the noise
        # residual.
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            dtype="float32",
            name="scheduler",
        )
        self.euler_step = EulerStep(dtype="float32", name="euler_step")
        self.image_rescaling = ImageRescaling(
            scale=self.vae.scale,
            offset=self.vae.shift,
            dtype=dtype,
            name="image_rescaling",
        )
        self.latent_rescaling = LatentRescaling(
            scale=self.vae.scale,
            offset=self.vae.shift,
            dtype=dtype,
            name="latent_rescaling",
        )

        # === Functional Model ===
        image_input = keras.Input(
            shape=image_shape,
            name="images",
        )
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
        embeddings = self.encode_text_step(token_ids, negative_token_ids)
        latents = self.encode_image_step(image_input)
        # Use `steps=0` to define the functional model.
        denoised_latents = self.denoise_step(
            latent_input,
            embeddings,
            0,
            num_step_input[0],
            guidance_scale_input[0],
        )
        images = self.decode_step(denoised_latents)
        inputs = {
            "images": image_input,
            "latents": latent_input,
            "clip_l_token_ids": clip_l_token_id_input,
            "clip_l_negative_token_ids": clip_l_negative_token_id_input,
            "clip_g_token_ids": clip_g_token_id_input,
            "clip_g_negative_token_ids": clip_g_negative_token_id_input,
            "num_steps": num_step_input,
            "guidance_scale": guidance_scale_input,
        }
        outputs = {
            "latents": latents,
            "images": images,
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
        self.mmdit_qk_norm = mmdit_qk_norm
        self.mmdit_dual_attention_indices = mmdit_dual_attention_indices
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.image_shape = image_shape

    @property
    def latent_shape(self):
        return (None,) + self._latent_shape

    @property
    def clip_hidden_dim(self):
        return self.clip_l.hidden_dim + self.clip_g.hidden_dim

    @property
    def t5_hidden_dim(self):
        return 4096 if self.t5 is None else self.t5.hidden_dim

    def encode_text_step(self, token_ids, negative_token_ids):
        clip_hidden_dim = self.clip_hidden_dim
        t5_hidden_dim = self.t5_hidden_dim

        def encode(token_ids):
            clip_l_outputs = self.clip_l(
                {"token_ids": token_ids["clip_l"]}, training=False
            )
            clip_g_outputs = self.clip_g(
                {"token_ids": token_ids["clip_g"]}, training=False
            )
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
            pooled_embeddings, embeddings = self.clip_concatenate(
                clip_l_projection,
                clip_g_projection,
                clip_l_outputs["intermediate_output"],
                clip_g_outputs["intermediate_output"],
                padding=t5_hidden_dim - clip_hidden_dim,
            )
            if self.t5 is not None:
                t5_outputs = self.t5(
                    {
                        "token_ids": token_ids["t5"],
                        "padding_mask": ops.ones_like(token_ids["t5"]),
                    },
                    training=False,
                )
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

    def encode_image_step(self, images):
        latents = self.vae.encode(images)
        return self.image_rescaling(latents)

    def add_noise_step(self, latents, noises, step, num_steps):
        return self.scheduler.add_noise(latents, noises, step, num_steps)

    def denoise_step(
        self,
        latents,
        embeddings,
        step,
        num_steps,
        guidance_scale=None,
    ):
        step = ops.convert_to_tensor(step)
        next_step = ops.add(step, 1)
        sigma, timestep = self.scheduler(step, num_steps)
        next_sigma, _ = self.scheduler(next_step, num_steps)

        # Concatenation for classifier-free guidance.
        if guidance_scale is not None:
            concated_latents, contexts, pooled_projs, timesteps = (
                self.cfg_concat(latents, *embeddings, timestep)
            )
        else:
            timesteps = ops.broadcast_to(timestep, ops.shape(latents)[:1])
            concated_latents = latents
            contexts = embeddings[0]
            pooled_projs = embeddings[2]

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
        if guidance_scale is not None:
            predicted_noise = self.cfg(predicted_noise, guidance_scale)

        # Euler step.
        return self.euler_step(latents, predicted_noise, sigma, next_sigma)

    def decode_step(self, latents):
        latents = self.latent_rescaling(latents)
        return self.vae.decode(latents, training=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mmdit_patch_size": self.mmdit_patch_size,
                "mmdit_hidden_dim": self.mmdit_hidden_dim,
                "mmdit_num_layers": self.mmdit_num_layers,
                "mmdit_num_heads": self.mmdit_num_heads,
                "mmdit_position_size": self.mmdit_position_size,
                "mmdit_qk_norm": self.mmdit_qk_norm,
                "mmdit_dual_attention_indices": (
                    self.mmdit_dual_attention_indices
                ),
                "vae": layers.serialize(self.vae),
                "clip_l": layers.serialize(self.clip_l),
                "clip_g": layers.serialize(self.clip_g),
                "t5": layers.serialize(self.t5),
                "latent_channels": self.latent_channels,
                "output_channels": self.output_channels,
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.shift,
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to text encoders if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["vae"]["config"]:
                config["vae"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["clip_l"]["config"]:
                config["clip_l"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["clip_g"]["config"]:
                config["clip_g"]["config"]["dtype"] = dtype_config
            if (
                config["t5"] is not None
                and "dtype" not in config["t5"]["config"]
            ):
                config["t5"]["config"]["dtype"] = dtype_config

        # We expect `vae`, `clip_l`, `clip_g` and/or `t5` to be instantiated.
        config["vae"] = layers.deserialize(
            config["vae"], custom_objects=custom_objects
        )
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

        # To maintain backward compatibility, we need to ensure that
        # `mmdit_qk_norm` and `mmdit_dual_attention_indices` is included in the
        # config.
        if "mmdit_qk_norm" not in config:
            config["mmdit_qk_norm"] = None
        if "mmdit_dual_attention_indices" not in config:
            config["mmdit_dual_attention_indices"] = None
        return cls(**config)
