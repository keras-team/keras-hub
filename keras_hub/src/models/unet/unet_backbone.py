import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.unet.unet_decoder import UNetDecoder
from keras_hub.src.models.unet.unet_encoder import UNetEncoder
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.UNetBackbone")
class UNetBackbone(Backbone):
    """UNet architecture for image segmentation.

    A Keras model implementing the UNet architecture described in [U-Net:
    Convolutional Networks for Biomedical Image Segmentation](
    https://arxiv.org/abs/1505.04597). UNet uses an encoder-decoder
    architecture with skip connections for precise image segmentation.

    This implementation supports:
    - Vanilla U-Net (original paper)
    - Modern U-Net with batch normalization and improved upsampling
    - ResUNet with residual connections
    - Attention U-Net with attention gates on skip connections
    - Custom pretrained backbones as encoder

    Args:
        backbone: optional `keras.Model`. A pretrained backbone to use as the
            encoder. If provided, the model will use this backbone's pyramid
            outputs as encoder features. If `None`, builds encoder from scratch.
            Defaults to `None`.
        depth: int. The depth of the U-Net architecture when building encoder
            from scratch, representing the number of downsampling/upsampling
            steps. Ignored if `backbone` is provided. Defaults to 4.
        filters: int. The number of filters in the first convolutional layer
            when building encoder from scratch. The number of filters doubles
            at each downsampling step and halves at each upsampling step.
            Ignored if `backbone` is provided. Defaults to 64.
        image_shape: optional shape tuple, defaults to `None`. If `None`,
            defaults to `(None, None, 3)` for `"channels_last"` data format
            or `(3, None, None)` for `"channels_first"` data format.
            Must have 3 channels in the correct position based on `data_format`.
            The dynamic spatial dimensions allow the model to accept inputs
            of any size.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. If not specified, uses the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. Defaults to `None`.
        use_batch_norm: bool. Whether to use batch normalization in the
            convolutional blocks. Defaults to False (as in original paper).
        use_dropout: bool. Whether to use dropout in the decoder path.
            Defaults to False.
        dropout_rate: float. Dropout rate if use_dropout is True.
            Defaults to 0.3.
        upsampling_strategy: str. Strategy for upsampling in decoder.
            Either `"transpose"` (uses Conv2DTranspose, original paper) or
            `"interpolation"` (uses UpSampling2D + Conv2D to avoid
            checkerboard artifacts). Defaults to `"transpose"`.
        use_residual: bool. Whether to add residual connections within
            convolutional blocks (ResUNet variant). Defaults to False.
        use_attention: bool. Whether to add attention gates to skip connections
            (Attention U-Net variant). Defaults to False.
        kernel_initializer: str or initializer. Defaults to "he_normal".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Vanilla U-Net from scratch
    model = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        image_shape=(None, None, 3),
    )

    # Modern U-Net with batch norm and better upsampling
    model = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        use_batch_norm=True,
        upsampling_strategy="interpolation",
    )

    # ResUNet with residual connections
    model = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        use_batch_norm=True,
        use_residual=True,
    )

    # Attention U-Net
    model = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        use_batch_norm=True,
        use_attention=True,
    )

    # Using pretrained backbone (e.g., ResNet50)
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )
    model = keras_hub.models.UNetBackbone(
        backbone=backbone,
        use_batch_norm=True,
        upsampling_strategy="interpolation",
    )

    # Can accept any image size
    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    output = model(images)
    ```
    """

    def __init__(
        self,
        backbone=None,
        depth=4,
        filters=64,
        image_shape=None,
        data_format=None,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.3,
        upsampling_strategy="transpose",
        use_residual=False,
        use_attention=False,
        kernel_initializer="he_normal",
        dtype=None,
        **kwargs,
    ):
        if upsampling_strategy not in ["transpose", "interpolation"]:
            raise ValueError(
                f"upsampling_strategy must be 'transpose' or 'interpolation'. "
                f"Received: {upsampling_strategy}"
            )

        data_format = standardize_data_format(data_format)

        if image_shape is None:
            if data_format == "channels_last":
                image_shape = (None, None, 3)
            else:
                image_shape = (3, None, None)

        self.backbone = backbone
        self.depth = depth
        self.filters = filters
        self.image_shape = image_shape
        self.data_format = data_format
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.upsampling_strategy = upsampling_strategy
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.kernel_initializer = kernel_initializer
        self._backbone_feature_names = None

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)

        # Build encoder using UNetEncoder
        self.encoder = UNetEncoder(
            backbone=backbone,
            depth=depth,
            filters=filters,
            image_shape=image_shape,
            data_format=data_format,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="encoder",
        )

        # Build decoder using UNetDecoder
        self.decoder = UNetDecoder(
            filters=filters if backbone is None else None,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            upsampling_strategy=upsampling_strategy,
            use_residual=use_residual,
            use_attention=use_attention,
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            dtype=dtype,
            name="decoder",
        )

        # Forward pass through encoder and decoder
        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output)

        # Handle additional upsampling for pretrained backbones
        x = decoder_output
        if backbone is not None:
            x = self._apply_final_upsampling(
                x,
                inputs,
                encoder_output,
                upsampling_strategy,
                kernel_initializer,
                data_format,
                dtype,
            )

        outputs = x

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

    def _apply_final_upsampling(
        self,
        x,
        inputs,
        encoder_output,
        upsampling_strategy,
        kernel_initializer,
        data_format,
        dtype,
    ):
        """Apply additional upsampling for pretrained backbones.

        Some backbones (like ResNet) downsample the image significantly
        (e.g., 32x).
        The default U-Net architecture might not upsample enough to match the
        original input size. This method handles any remaining upsampling
        needed.
        """
        # Calculate current downsampling factor
        input_size = (
            inputs.shape[1]
            if data_format == "channels_last"
            else inputs.shape[2]
        )
        x_size = x.shape[1] if data_format == "channels_last" else x.shape[2]

        if input_size is None or x_size is None:
            # Cannot determine if upsampling is needed dynamically from shapes
            # alone.
            # Fallback to checking encoder metadata to see if first skip
            # connection was downsampled.
            if (
                hasattr(self.encoder, "_backbone_feature_names")
                and self.encoder._backbone_feature_names
            ):
                first_layer_name = self.encoder._backbone_feature_names[
                    0
                ].lower()
                # Common patterns where the first extracted feature is already
                # downsampled relative to the input image (e.g., initial 7x7
                # stride-2 conv in ResNet).
                first_skip_downsampled = any(
                    pattern in first_layer_name
                    for pattern in ["conv1", "block_1", "stem", "pool"]
                )
                if first_skip_downsampled:
                    # We assume a factor of 2 if we can't calculate it.
                    # This covers most standard backbones (ResNet,
                    # MobileNet, EfficientNet).
                    factor = 2

                    # Apply upsampling
                    if upsampling_strategy == "transpose":
                        x = keras.layers.Conv2DTranspose(
                            x.shape[-1]
                            if data_format == "channels_last"
                            else x.shape[0],
                            kernel_size=factor,
                            strides=factor,
                            padding="same",
                            kernel_initializer=kernel_initializer,
                            data_format=data_format,
                            dtype=dtype,
                            name="final_upsample",
                        )(x)
                    else:
                        x = keras.layers.UpSampling2D(
                            size=factor,
                            data_format=data_format,
                            interpolation="bilinear",
                            dtype=dtype,
                            name="final_upsample",
                        )(x)
                        x = keras.layers.Conv2D(
                            x.shape[-1]
                            if data_format == "channels_last"
                            else x.shape[0],
                            kernel_size=(3, 3),
                            padding="same",
                            kernel_initializer=kernel_initializer,
                            data_format=data_format,
                            dtype=dtype,
                            name="final_upsample_conv",
                        )(x)
            return x

        if x_size >= input_size:
            return x

        # Needed upsampling factor
        factor = input_size // x_size

        if factor <= 1:
            return x

        # Final upsampling block
        filters = x.shape[-1] if data_format == "channels_last" else x.shape[0]

        if upsampling_strategy == "transpose":
            x = keras.layers.Conv2DTranspose(
                filters,
                kernel_size=factor,
                strides=factor,
                padding="same",
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name="final_upsample",
            )(x)
        else:
            x = keras.layers.UpSampling2D(
                size=factor,
                data_format=data_format,
                interpolation="bilinear",
                dtype=dtype,
                name="final_upsample",
            )(x)
            x = keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name="final_upsample_conv",
            )(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone)
                if self.backbone is not None
                else None,
                "depth": self.depth,
                "filters": self.filters,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
                "use_batch_norm": self.use_batch_norm,
                "use_dropout": self.use_dropout,
                "dropout_rate": self.dropout_rate,
                "upsampling_strategy": self.upsampling_strategy,
                "use_residual": self.use_residual,
                "use_attention": self.use_attention,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        return cls(**config)
