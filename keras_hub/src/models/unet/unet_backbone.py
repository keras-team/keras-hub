import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.UNetBackbone")
class UNetBackbone(Backbone):
    """UNet architecture for image segmentation.

    A Keras model implementing the UNet architecture described in [U-Net:
    Convolutional Networks for Biomedical Image Segmentation](
    https://arxiv.org/abs/1505.04597). UNet uses an encoder-decoder
    architecture with skip connections for precise image segmentation.

    Args:
        depth: int. The depth of the U-Net architecture, representing the
            number of downsampling/upsampling steps. Defaults to 4.
        filters: int. The number of filters in the first convolutional layer.
            The number of filters doubles at each downsampling step and halves
            at each upsampling step. Defaults to 64.
        image_shape: optional shape tuple, defaults to (None, None, 3).
            Must have 3 channels. The dynamic spatial dimensions allow the
            model to accept inputs of any size.
        use_batch_norm: bool. Whether to use batch normalization in the
            convolutional blocks. Defaults to False (as in original paper).
        use_dropout: bool. Whether to use dropout in the decoder path.
            Defaults to False.
        dropout_rate: float. Dropout rate if use_dropout is True.
            Defaults to 0.3.
        kernel_initializer: str or initializer. Defaults to "he_normal".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Randomly initialized UNet backbone with default config.
    model = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        image_shape=(None, None, 3),
    )

    # Can accept any image size
    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    output = model(images)

    # Different size
    images = np.random.uniform(0, 1, size=(1, 512, 512, 3))
    output = model(images)
    ```
    """

    def __init__(
        self,
        depth=4,
        filters=64,
        image_shape=(None, None, 3),
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.3,
        kernel_initializer="he_normal",
        dtype=None,
        **kwargs,
    ):
        if image_shape[-1] != 3:
            raise ValueError(
                f"image_shape must have 3 channels. Received: {image_shape}"
            )

        data_format = keras.config.image_data_format()

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)
        x = inputs

        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder (downsampling path)
        for i in range(depth):
            num_filters = filters * (2**i)
            x = self._conv_block(
                x,
                num_filters,
                use_batch_norm=use_batch_norm,
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name=f"encoder_block_{i}",
            )
            encoder_outputs.append(x)
            if i < depth - 1:  # Don't pool at the bottom
                x = keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    data_format=data_format,
                    dtype=dtype,
                    name=f"encoder_pool_{i}",
                )(x)

        # Decoder (upsampling path)
        for i in range(depth - 2, -1, -1):
            num_filters = filters * (2**i)
            x = keras.layers.Conv2DTranspose(
                num_filters,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="same",
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name=f"decoder_upsample_{i}",
            )(x)

            # Skip connection from encoder
            skip_connection = encoder_outputs[i]
            x = keras.layers.Concatenate(
                axis=-1 if data_format == "channels_last" else 1,
                dtype=dtype,
                name=f"decoder_concat_{i}",
            )([x, skip_connection])

            if use_dropout and i > 0:
                x = keras.layers.Dropout(
                    dropout_rate, dtype=dtype, name=f"decoder_dropout_{i}"
                )(x)

            x = self._conv_block(
                x,
                num_filters,
                use_batch_norm=use_batch_norm,
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )

        outputs = x

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.depth = depth
        self.filters = filters
        self.image_shape = image_shape
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

    def _conv_block(
        self,
        x,
        filters,
        use_batch_norm=False,
        kernel_initializer="he_normal",
        data_format="channels_last",
        dtype=None,
        name="conv_block",
    ):
        """Double convolution block used in UNet.

        Args:
            x: Input tensor.
            filters: Number of filters for the convolution layers.
            use_batch_norm: Whether to use batch normalization.
            kernel_initializer: Initializer for the kernel weights.
            data_format: Data format for the layers.
            dtype: Data type for the layers.
            name: Name prefix for the layers.

        Returns:
            Output tensor after the double convolution.
        """
        x = keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv1",
        )(x)
        if use_batch_norm:
            bn_axis = -1 if data_format == "channels_last" else 1
            x = keras.layers.BatchNormalization(
                axis=bn_axis, dtype=dtype, name=f"{name}_bn1"
            )(x)
        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu1")(
            x
        )

        x = keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv2",
        )(x)
        if use_batch_norm:
            bn_axis = -1 if data_format == "channels_last" else 1
            x = keras.layers.BatchNormalization(
                axis=bn_axis, dtype=dtype, name=f"{name}_bn2"
            )(x)
        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu2")(
            x
        )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "filters": self.filters,
                "image_shape": self.image_shape,
                "use_batch_norm": self.use_batch_norm,
                "use_dropout": self.use_dropout,
                "dropout_rate": self.dropout_rate,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
