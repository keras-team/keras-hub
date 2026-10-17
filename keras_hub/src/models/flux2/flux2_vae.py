import keras
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.vae.vae_layers import (
    DiagonalGaussianDistributionSampler,
)
from keras_hub.src.models.vae.vae_layers import VAEDecoder
from keras_hub.src.models.vae.vae_layers import VAEEncoder


def patchify_latents(latents, patch_size=2):
    """`(B, H, W, C) -> (B, H/p, W/p, C*p*p)` with 2D patches as channels.

    The channel index is `c*p*p + dy*p + dx`. This is the channels-last
    equivalent of `Flux2Pipeline._patchify_latents`
    (`(B, C, H, W) -> (B, C*4, H/2, W/2)`). The channel order matters: the
    VAE's BatchNorm statistics are stored per patchified channel.
    """
    shape = ops.shape(latents)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    p = patch_size
    x = ops.reshape(latents, (batch, height // p, p, width // p, p, channels))
    x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    return ops.reshape(x, (batch, height // p, width // p, channels * p * p))


def unpatchify_latents(latents, patch_size=2):
    """Inverse of `patchify_latents`."""
    shape = ops.shape(latents)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    p = patch_size
    x = ops.reshape(latents, (batch, height, width, channels // (p * p), p, p))
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
    return ops.reshape(x, (batch, height * p, width * p, channels // (p * p)))


class Flux2LatentNormalization(keras.layers.Layer):
    """Maps between VAE latents and the FLUX.2 transformer's latent space.

    FLUX.2 normalizes latents with the running statistics of an affine-free
    `BatchNorm2d` over 2x2-patchified VAE latents (`AutoencoderKLFlux2.bn`),
    instead of FLUX.1's scalar `scaling_factor` / `shift_factor`.

    Called forward, it patchifies `(B, H, W, C)` VAE latents and standardizes
    them to the transformer's `(B, H/2, W/2, 4C)` space. Called with
    `inverse=True` it undoes that, for decoding.

    Args:
        latent_channels: int. The number of VAE latent channels `C`.
        patch_size: int. The patch size. Defaults to `2`.
        epsilon: float. The BatchNorm epsilon. Defaults to `1e-4`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, latent_channels, patch_size=2, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        self.epsilon = float(epsilon)
        num_channels = self.latent_channels * self.patch_size**2
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(num_channels,),
            initializer="zeros",
            trainable=False,
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(num_channels,),
            initializer="ones",
            trainable=False,
        )

    def call(self, inputs, inverse=False):
        dtype = self.compute_dtype
        mean = ops.cast(self.moving_mean, dtype)
        std = ops.sqrt(ops.cast(self.moving_variance, dtype) + self.epsilon)
        x = ops.cast(inputs, dtype)
        if inverse:
            return unpatchify_latents(x * std + mean, self.patch_size)
        return (patchify_latents(x, self.patch_size) - mean) / std

    def compute_output_spec(self, inputs, inverse=False):
        batch, height, width, channels = inputs.shape
        p = self.patch_size
        if inverse:
            height = None if height is None else height * p
            width = None if width is None else width * p
            channels = channels // (p * p)
        else:
            height = None if height is None else height // p
            width = None if width is None else width // p
            channels = channels * p * p
        return keras.KerasTensor(
            (batch, height, width, channels), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_channels": self.latent_channels,
                "patch_size": self.patch_size,
                "epsilon": self.epsilon,
            }
        )
        return config


class Flux2VAE(Backbone):
    """The FLUX.2 autoencoder (`AutoencoderKLFlux2`).

    The encoder and decoder are the standard latent-diffusion blocks shared
    with Stable Diffusion 3 and FLUX.1 (`VAEEncoder` / `VAEDecoder`). FLUX.2
    adds:

    - 32 latent channels (16 in FLUX.1);
    - `quant_conv` / `post_quant_conv` 1x1 convolutions around the latent;
    - latent normalization with BatchNorm statistics over 2x2-patchified
      latents (`Flux2LatentNormalization`).

    `encode()` and `decode()` work in the transformer's latent space:
    `(B, H/16, W/16, 4 * latent_channels)` for `(B, H, W, 3)` images, so a
    denoised latent grid can be decoded directly.

    Args:
        encoder_num_filters: list of ints. The filters of each encoder stage.
        encoder_num_blocks: list of ints. The ResNet blocks of each encoder
            stage.
        decoder_num_filters: list of ints. The filters of each decoder stage.
        decoder_num_blocks: list of ints. The ResNet blocks of each decoder
            stage.
        latent_channels: int. The number of latent channels. Defaults to `32`.
        patch_size: int. The latent patch size. Defaults to `2`.
        batch_norm_epsilon: float. Defaults to `1e-4`.
        sampler_method: str. `"mode"` (the posterior mean, as the FLUX.2
            pipelines use) or `"sample"`. Defaults to `"mode"`.
        input_channels: int. Defaults to `3`.
        output_channels: int. Defaults to `3`.
        dtype: `None`, str or `keras.mixed_precision.DTypePolicy`.
    """

    def __init__(
        self,
        encoder_num_filters,
        encoder_num_blocks,
        decoder_num_filters,
        decoder_num_blocks,
        latent_channels=32,
        patch_size=2,
        batch_norm_epsilon=1e-4,
        sampler_method="mode",
        input_channels=3,
        output_channels=3,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.encoder = VAEEncoder(
            encoder_num_filters,
            encoder_num_blocks,
            output_channels=2 * latent_channels,
            data_format="channels_last",
            dtype=dtype,
            name="encoder",
        )
        self.quant_conv = keras.layers.Conv2D(
            2 * latent_channels, 1, dtype=dtype, name="quant_conv"
        )
        self.distribution_sampler = DiagonalGaussianDistributionSampler(
            method=sampler_method,
            axis=-1,
            dtype=dtype,
            name="distribution_sampler",
        )
        self.latent_normalization = Flux2LatentNormalization(
            latent_channels,
            patch_size=patch_size,
            epsilon=batch_norm_epsilon,
            dtype=dtype,
            name="latent_normalization",
        )
        self.post_quant_conv = keras.layers.Conv2D(
            latent_channels, 1, dtype=dtype, name="post_quant_conv"
        )
        self.decoder = VAEDecoder(
            decoder_num_filters,
            decoder_num_blocks,
            output_channels=output_channels,
            data_format="channels_last",
            dtype=dtype,
            name="decoder",
        )

        # === Functional Model ===
        image_input = keras.Input(shape=(None, None, input_channels))
        latents = self.encode(image_input)
        images = self.decode(latents)
        super().__init__(
            inputs=image_input, outputs=images, dtype=dtype, **kwargs
        )

        # === Config ===
        self.encoder_num_filters = encoder_num_filters
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_filters = decoder_num_filters
        self.decoder_num_blocks = decoder_num_blocks
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.batch_norm_epsilon = batch_norm_epsilon
        self.sampler_method = sampler_method
        self.input_channels = input_channels
        self.output_channels = output_channels

    def encode(self, images, **kwargs):
        """Images in `[-1, 1]` to normalized, patchified latents."""
        x = self.encoder(images, **kwargs)
        x = self.quant_conv(x)
        x = self.distribution_sampler(x)
        return self.latent_normalization(x)

    def decode(self, latents, **kwargs):
        """Normalized, patchified latents to images in `[-1, 1]`."""
        x = self.latent_normalization(latents, inverse=True)
        x = self.post_quant_conv(x)
        return self.decoder(x, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_num_filters": self.encoder_num_filters,
                "encoder_num_blocks": self.encoder_num_blocks,
                "decoder_num_filters": self.decoder_num_filters,
                "decoder_num_blocks": self.decoder_num_blocks,
                "latent_channels": self.latent_channels,
                "patch_size": self.patch_size,
                "batch_norm_epsilon": self.batch_norm_epsilon,
                "sampler_method": self.sampler_method,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
            }
        )
        return config
