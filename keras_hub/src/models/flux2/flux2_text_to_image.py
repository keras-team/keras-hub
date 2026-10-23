import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_scheduler import Flux2FlowMatchScheduler
from keras_hub.src.models.flux2.flux2_text_to_image_preprocessor import (
    Flux2TextToImagePreprocessor,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    ClassifierFreeGuidance,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    EulerStep,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    TimestepBroadcastTo,
)
from keras_hub.src.models.text_to_image import TextToImage


class Flux2PositionIds(layers.Layer):
    """Builds FLUX.2's 4-axis `(t, h, w, l)` RoPE positions for text-to-image.

    Text token `i` is at `(0, 0, 0, i)` and latent token `(row, col)` at
    `(0, row, col, 0)`, as in `Flux2KleinPipeline._prepare_text_ids` and
    `_prepare_latent_ids`. Latent tokens are in row-major order.

    Args:
        latent_height: int. The height of the latent token grid.
        latent_width: int. The width of the latent token grid.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, latent_height, latent_width, **kwargs):
        super().__init__(**kwargs)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)

    def call(self, text, image):
        text_shape = ops.shape(text)
        index = ops.cast(ops.arange(text_shape[1]), "float32")
        zeros = ops.zeros_like(index)
        text_ids = ops.stack([zeros, zeros, zeros, index], axis=-1)
        text_ids = ops.broadcast_to(
            text_ids[None], (text_shape[0], text_shape[1], 4)
        )

        rows = ops.repeat(ops.arange(self.latent_height), self.latent_width)
        cols = ops.tile(ops.arange(self.latent_width), [self.latent_height])
        rows = ops.cast(rows, "float32")
        cols = ops.cast(cols, "float32")
        zeros = ops.zeros_like(rows)
        image_ids = ops.stack([zeros, rows, cols, zeros], axis=-1)
        num_tokens = self.latent_height * self.latent_width
        image_ids = ops.broadcast_to(
            image_ids[None], (ops.shape(image)[0], num_tokens, 4)
        )
        return text_ids, image_ids

    def compute_output_spec(self, text, image):
        num_tokens = self.latent_height * self.latent_width
        return (
            keras.KerasTensor((text.shape[0], text.shape[1], 4), "float32"),
            keras.KerasTensor((image.shape[0], num_tokens, 4), "float32"),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_height": self.latent_height,
                "latent_width": self.latent_width,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Flux2TextToImage")
class Flux2TextToImage(TextToImage):
    """An end-to-end FLUX.2 [klein] model for text-to-image generation.

    The pipeline is:

    1. `Flux2TextToImagePreprocessor` wraps the prompt in the Qwen3 chat
       template and tokenizes it to 512 tokens.
    2. The Qwen3 `text_encoder` runs up to its deepest requested layer, and
       the Hugging Face `hidden_states` listed in `text_encoder_layers` (by
       default `(9, 18, 27)`: the outputs of the 9th, 18th and 27th decoder
       blocks) are concatenated along the feature axis: `3 * 2560 = 7680`
       features for klein 4B.
    3. Starting from Gaussian noise in the transformer's latent space, the
       `Flux2Backbone` velocity is integrated with Euler steps over the
       FLUX.2 flow-matching schedule (`Flux2FlowMatchScheduler`).
    4. `Flux2VAE` decodes the latents to an image.

    FLUX.2 [klein] is step-distilled (`is_distilled=True`): 4 steps, no
    classifier-free guidance, and no negative prompts. The non-distilled
    "base" variants use classifier-free guidance with negative prompts
    (`is_distilled=False`).

    Args:
        backbone: A `keras_hub.models.Flux2Backbone` instance.
        preprocessor: A `keras_hub.models.Flux2TextToImagePreprocessor`
            instance.
        text_encoder: A `keras_hub.models.Qwen3Backbone` instance.
        vae: A `Flux2VAE` instance.
        text_encoder_layers: tuple of ints. The text encoder layers whose
            outputs are concatenated, as Hugging Face `hidden_states` indices
            (`0` is the embedding output). Defaults to `(9, 18, 27)`.
        image_shape: tuple. `(height, width, 3)` of the generated images.
            Both sides must be divisible by the VAE downsampling factor times
            the patch size (`16`). Defaults to `(1024, 1024, 3)`.
        is_distilled: bool. Whether the model is step-distilled (no
            classifier-free guidance). Defaults to `True`.

    Examples:
    ```python
    text_to_image = keras_hub.models.Flux2TextToImage.from_preset(
        "flux2_klein_4b", image_shape=(512, 512, 3)
    )
    text_to_image.generate("A cat holding a sign that says hello world")

    # Batched prompts.
    text_to_image.generate(["a red bicycle", "a blue house"])
    ```
    """

    backbone_cls = Flux2Backbone
    preprocessor_cls = Flux2TextToImagePreprocessor

    def __init__(
        self,
        backbone,
        preprocessor,
        text_encoder,
        vae,
        text_encoder_layers=(9, 18, 27),
        image_shape=(1024, 1024, 3),
        is_distilled=True,
        **kwargs,
    ):
        text_encoder_layers = tuple(int(i) for i in text_encoder_layers)
        if backbone.guidance_embed:
            raise ValueError(
                "`Flux2TextToImage` supports FLUX.2 [klein], which has no "
                "guidance embedding. Received a backbone with "
                "`guidance_embed=True` (FLUX.2 [dev])."
            )
        if not text_encoder_layers or not all(
            0 <= i <= text_encoder.num_layers for i in text_encoder_layers
        ):
            raise ValueError(
                "`text_encoder_layers` must be Hugging Face `hidden_states` "
                f"indices in [0, {text_encoder.num_layers}]. Received: "
                f"{text_encoder_layers}"
            )
        text_dim = len(text_encoder_layers) * text_encoder.hidden_dim
        if backbone.text_shape[-1] != text_dim:
            raise ValueError(
                f"The backbone expects {backbone.text_shape[-1]} text "
                f"features, but {len(text_encoder_layers)} text encoder layers "
                f"of width {text_encoder.hidden_dim} give {text_dim}."
            )
        latent_channels = vae.latent_channels * vae.patch_size**2
        if backbone.input_channels != latent_channels:
            raise ValueError(
                f"The backbone expects {backbone.input_channels} latent "
                f"channels, but the VAE produces {latent_channels}."
            )
        factor = 2 ** (len(vae.decoder_num_filters) - 1) * vae.patch_size
        height, width = int(image_shape[0]), int(image_shape[1])
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"`image_shape` height and width must be divisible by "
                f"{factor}. Received: image_shape={image_shape}"
            )
        latent_height, latent_width = height // factor, width // factor
        self._latent_shape = (latent_height, latent_width, latent_channels)
        self.text_encoder_layers = text_encoder_layers
        self.is_distilled = bool(is_distilled)

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.text_encoder = text_encoder
        self.vae = vae
        self.position_ids = Flux2PositionIds(
            latent_height, latent_width, dtype="float32", name="position_ids"
        )
        # The schedule and the Euler update stay in float32, as in the
        # reference scheduler.
        self.scheduler = Flux2FlowMatchScheduler(
            latent_height * latent_width, dtype="float32", name="scheduler"
        )
        self.timestep_broadcast_to = TimestepBroadcastTo(
            dtype="float32", name="timestep_broadcast_to"
        )
        self.cfg = ClassifierFreeGuidance(
            dtype=backbone.dtype_policy, name="classifier_free_guidance"
        )
        self.euler_step = EulerStep(dtype="float32", name="euler_step")

        # === Functional Model ===
        # One denoising step plus decoding, so every component is part of
        # the graph. `generate()` runs the full loop.
        latent_input = keras.Input(shape=self._latent_shape, name="latents")
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        num_step_input = keras.Input(shape=(), dtype="int32", name="num_steps")
        inputs = {
            "latents": latent_input,
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "num_steps": num_step_input,
        }
        token_ids = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        negative_token_ids = None
        guidance_scale = None
        if not self.is_distilled:
            negative_token_id_input = keras.Input(
                shape=(None,), dtype="int32", name="negative_token_ids"
            )
            negative_padding_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="negative_padding_mask"
            )
            guidance_scale_input = keras.Input(
                shape=(), dtype="float32", name="guidance_scale"
            )
            inputs["negative_token_ids"] = negative_token_id_input
            inputs["negative_padding_mask"] = negative_padding_mask_input
            inputs["guidance_scale"] = guidance_scale_input
            negative_token_ids = {
                "token_ids": negative_token_id_input,
                "padding_mask": negative_padding_mask_input,
            }
            guidance_scale = guidance_scale_input[0]

        embeddings = self.encode_text_step(token_ids, negative_token_ids)
        latents = self.denoise_step(
            latent_input, embeddings, 0, num_step_input[0], guidance_scale
        )
        images = self.decode_step(latents)
        super().__init__(inputs=inputs, outputs=images, **kwargs)

        # === Config ===
        self.image_shape = tuple(image_shape)

    @property
    def latent_shape(self):
        return (None,) + self._latent_shape

    @property
    def support_negative_prompts(self):
        # Step-distilled models do not use classifier-free guidance.
        return not self.is_distilled

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Currently, `fit` is not supported for `Flux2TextToImage`."
        )

    def encode_text_step(self, token_ids, negative_token_ids=None):
        """Encode token ids to the transformer's text conditioning.

        Runs the text encoder up to its deepest requested layer and
        concatenates the requested layer outputs along the feature axis.
        Indices follow Hugging Face `output_hidden_states`: `0` is the
        embedding output, `i` the output of decoder block `i - 1`, and
        `num_layers` the final, normalized output.

        Args:
            token_ids: A dict with `"token_ids"` and `"padding_mask"`.
            negative_token_ids: Optional dict, as `token_ids`, for
                classifier-free guidance.

        Returns:
            A `(positive, negative)` tuple of `(batch, sequence, features)`
            tensors. `negative` is `None` if `negative_token_ids` is.
        """

        def encode(inputs):
            x = self.text_encoder.token_embedding(inputs["token_ids"])
            hidden_states = {0: x}
            for i in range(max(self.text_encoder_layers)):
                x = self.text_encoder.transformer_layers[i](
                    x, decoder_padding_mask=inputs["padding_mask"]
                )
                # `hidden_states[i + 1]` is the output of decoder block `i`.
                hidden_states[i + 1] = x
            num_layers = self.text_encoder.num_layers
            if num_layers in hidden_states:
                # Hugging Face normalizes the last hidden state.
                hidden_states[num_layers] = self.text_encoder.layer_norm(x)
            return ops.concatenate(
                [hidden_states[i] for i in self.text_encoder_layers], axis=-1
            )

        positive = encode(token_ids)
        negative = None
        if negative_token_ids is not None:
            negative = encode(negative_token_ids)
        return positive, negative

    def denoise_step(
        self, latents, embeddings, step, num_steps, guidance_scale=None
    ):
        """One Euler step of the flow-matching ODE.

        Args:
            latents: A `(batch, latent_height, latent_width, channels)`
                tensor in the transformer's latent space.
            embeddings: The `(positive, negative)` tuple returned by
                `encode_text_step`.
            step: int or int scalar tensor. The current step.
            num_steps: int or int scalar tensor. The total number of steps.
            guidance_scale: Optional float scalar. Enables classifier-free
                guidance with the negative embeddings when set.
        """
        if not keras.utils.is_keras_tensor(step):
            step = ops.convert_to_tensor(step)
        if not keras.utils.is_keras_tensor(num_steps):
            num_steps = ops.convert_to_tensor(num_steps)
        latent_height, latent_width, channels = self._latent_shape
        sigma = self.scheduler(step, num_steps)
        next_sigma = self.scheduler(ops.add(step, 1), num_steps)

        positive, negative = embeddings
        packed = ops.reshape(
            latents, (-1, latent_height * latent_width, channels)
        )
        if guidance_scale is not None:
            image = ops.concatenate([packed, packed], axis=0)
            text = ops.concatenate([positive, negative], axis=0)
        else:
            image, text = packed, positive
        text_ids, image_ids = self.position_ids(text, image)
        timesteps = self.timestep_broadcast_to(image, sigma)

        velocity = self.backbone(
            {
                "image": image,
                "image_ids": image_ids,
                "text": text,
                "text_ids": text_ids,
                "timesteps": timesteps,
            },
            training=False,
        )
        if guidance_scale is not None:
            velocity = self.cfg(velocity, guidance_scale)
        velocity = ops.reshape(
            velocity, (-1, latent_height, latent_width, channels)
        )
        return self.euler_step(latents, velocity, sigma, next_sigma)

    def decode_step(self, latents):
        """Decode latents to images in `[-1, 1]`."""
        return self.vae.decode(latents, training=False)

    def generate_step(self, latents, token_ids, num_steps, guidance_scale):
        """A compilable generation function for a batch of inputs.

        Args:
            latents: A `(batch, latent_height, latent_width, channels)`
                tensor of initial Gaussian noise.
            token_ids: The preprocessed prompts, a dict with `"token_ids"`
                and `"padding_mask"`, or a `(prompts, negative_prompts)`
                pair of such dicts when `support_negative_prompts`.
            num_steps: int scalar. The number of Euler steps.
            guidance_scale: Optional float scalar for classifier-free
                guidance. Ignored for step-distilled models.
        """
        negative_token_ids = None
        if self.support_negative_prompts:
            token_ids, negative_token_ids = token_ids
        else:
            guidance_scale = None
        if guidance_scale is None:
            negative_token_ids = None

        embeddings = self.encode_text_step(token_ids, negative_token_ids)

        def body_fun(step, latents):
            return self.denoise_step(
                latents, embeddings, step, num_steps, guidance_scale
            )

        latents = ops.fori_loop(0, num_steps, body_fun, latents)
        return self.decode_step(latents)

    def generate(
        self,
        inputs,
        num_steps=None,
        guidance_scale=None,
        seed=None,
    ):
        """Generate images from prompts.

        Args:
            inputs: A string, a list of strings, or (for non-distilled
                models) a dict with `"prompts"` and `"negative_prompts"`.
            num_steps: int. Defaults to `4` for step-distilled models and
                `50` otherwise.
            guidance_scale: float. The classifier-free guidance scale.
                Ignored for step-distilled models. Defaults to `4.0` for
                non-distilled models.
            seed: optional int. Used as a random seed.
        """
        if num_steps is None:
            num_steps = 4 if self.is_distilled else 50
        if self.is_distilled:
            guidance_scale = None
        elif guidance_scale is None:
            guidance_scale = 4.0
        return super().generate(
            inputs,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_encoder": layers.serialize(self.text_encoder),
                "vae": layers.serialize(self.vae),
                "text_encoder_layers": list(self.text_encoder_layers),
                "image_shape": list(self.image_shape),
                "is_distilled": self.is_distilled,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        backbone = config.get("backbone")
        if isinstance(backbone, dict):
            backbone_config = backbone.get("config", {})
            image_shape = backbone_config.get("image_shape")
            # `from_preset(..., image_shape=(height, width, 3))` merges the
            # image size into the backbone config, where Stable Diffusion 3
            # keeps it. Here it is the task's output size; the backbone's
            # `image_shape` is the `(tokens, channels)` shape of the packed
            # latents and is never 3D.
            if image_shape is not None and len(image_shape) == 3:
                config["image_shape"] = image_shape
                latent_shape = [None, backbone_config["input_channels"]]
                config["backbone"] = {
                    **backbone,
                    "config": {**backbone_config, "image_shape": latent_shape},
                }
        for key in ("text_encoder", "vae"):
            if key in config and isinstance(config[key], dict):
                config[key] = layers.deserialize(config[key])
        return super().from_config(config)
