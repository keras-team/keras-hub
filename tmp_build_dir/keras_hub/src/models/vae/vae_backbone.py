import keras

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.vae.vae_layers import (
    DiagonalGaussianDistributionSampler,
)
from keras_hub.src.models.vae.vae_layers import VAEDecoder
from keras_hub.src.models.vae.vae_layers import VAEEncoder
from keras_hub.src.utils.keras_utils import standardize_data_format


class VAEBackbone(Backbone):
    """Variational Autoencoder(VAE) backbone used in latent diffusion models.

    When encoding, this model generates mean and log variance of the input
    images. When decoding, it reconstructs images from the latent space.

    Args:
        encoder_num_filters: list of ints. The number of filters for each
            block in encoder.
        encoder_num_blocks: list of ints. The number of blocks for each block in
            encoder.
        decoder_num_filters: list of ints. The number of filters for each
            block in decoder.
        decoder_num_blocks: list of ints. The number of blocks for each block in
            decoder.
        sampler_method: str. The method of the sampler for the intermediate
            output. Available methods are `"sample"` and `"mode"`. `"sample"`
            draws from the distribution using both the mean and log variance.
            `"mode"` draws from the distribution using the mean only. Defaults
            to `sample`.
        input_channels: int. The number of channels in the input.
        sample_channels: int. The number of channels in the sample. Typically,
            this indicates the intermediate output of VAE, which is mean and
            log variance.
        output_channels: int. The number of channels in the output.
        scale: float. The scaling factor applied to the latent space to ensure
            it has unit variance during training of the diffusion model.
            Defaults to `1.5305`, which is the value used in Stable Diffusion 3.
        shift: float. The shift factor applied to the latent space to ensure it
            has zero mean during training of the diffusion model. Defaults to
            `0.0609`, which is the value used in Stable Diffusion 3.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Example:
    ```Python
    backbone = VAEBackbone(
        encoder_num_filters=[32, 32, 32, 32],
        encoder_num_blocks=[1, 1, 1, 1],
        decoder_num_filters=[32, 32, 32, 32],
        decoder_num_blocks=[1, 1, 1, 1],
    )
    input_data = ops.ones((2, self.height, self.width, 3))
    output = backbone(input_data)
    ```
    """

    def __init__(
        self,
        encoder_num_filters,
        encoder_num_blocks,
        decoder_num_filters,
        decoder_num_blocks,
        sampler_method="sample",
        input_channels=3,
        sample_channels=32,
        output_channels=3,
        scale=1.5305,
        shift=0.0609,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            image_shape = (None, None, input_channels)
            channel_axis = -1
        else:
            image_shape = (input_channels, None, None)
            channel_axis = 1

        # === Layers ===
        self.encoder = VAEEncoder(
            encoder_num_filters,
            encoder_num_blocks,
            output_channels=sample_channels,
            data_format=data_format,
            dtype=dtype,
            name="encoder",
        )
        # Use `sample()` to define the functional model.
        self.distribution_sampler = DiagonalGaussianDistributionSampler(
            method=sampler_method,
            axis=channel_axis,
            dtype=dtype,
            name="distribution_sampler",
        )
        self.decoder = VAEDecoder(
            decoder_num_filters,
            decoder_num_blocks,
            output_channels=output_channels,
            data_format=data_format,
            dtype=dtype,
            name="decoder",
        )

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape)
        sample = self.encoder(image_input)
        latent = self.distribution_sampler(sample)
        image_output = self.decoder(latent)
        super().__init__(
            inputs=image_input,
            outputs=image_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.encoder_num_filters = encoder_num_filters
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_filters = decoder_num_filters
        self.decoder_num_blocks = decoder_num_blocks
        self.sampler_method = sampler_method
        self.input_channels = input_channels
        self.sample_channels = sample_channels
        self.output_channels = output_channels
        self._scale = scale
        self._shift = shift

    @property
    def scale(self):
        """The scaling factor for the latent space.

        This is used to scale the latent space to have unit variance when
        training the diffusion model.
        """
        return self._scale

    @property
    def shift(self):
        """The shift factor for the latent space.

        This is used to shift the latent space to have zero mean when
        training the diffusion model.
        """
        return self._shift

    def encode(self, inputs, **kwargs):
        """Encode the input images into latent space."""
        sample = self.encoder(inputs, **kwargs)
        return self.distribution_sampler(sample)

    def decode(self, inputs, **kwargs):
        """Decode the input latent space into images."""
        return self.decoder(inputs, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_num_filters": self.encoder_num_filters,
                "encoder_num_blocks": self.encoder_num_blocks,
                "decoder_num_filters": self.decoder_num_filters,
                "decoder_num_blocks": self.decoder_num_blocks,
                "sampler_method": self.sampler_method,
                "input_channels": self.input_channels,
                "sample_channels": self.sample_channels,
                "output_channels": self.output_channels,
                "scale": self.scale,
                "shift": self.shift,
            }
        )
        return config
