import functools

from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.XceptionBackbone")
class XceptionBackbone(Backbone):
    """Xception core network with hyperparameters.

    This class implements a Xception backbone as described in
    [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357).

    Most users will want the pretrained presets available with this model. If
    you are creating a custom backbone, this model provides customizability
    through the `stackwise_conv_filters` and `stackwise_pooling` arguments. This
    backbone assumes the same basic structure as the original Xception mode:
    * Residuals and pre-activation everywhere but the first and last block.
    * Conv layers for the first block only, separable conv layers elsewhere.

    Args:
        stackwise_conv_filters: list of list of ints. Each outermost list
            entry represents a block, and each innermost list entry a conv
            layer. The integer value specifies the number of filters for the
            conv layer.
        stackwise_pooling: list of bools. A list of booleans per block, where
            each entry is true if the block should includes a max pooling layer
            and false if it should not.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. If unspecified, the Keras default will be used.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Examples:
    ```python
    input_data = np.random.uniform(0, 1, size=(2, 224, 224, 3))

    # Pretrained Xception backbone.
    model = keras_hub.models.Backbone.from_preset("xception_41_imagenet")
    model(input_data)

    # Randomly initialized Xception backbone with a custom config.
    model = keras_hub.models.XceptionBackbone(
        stackwise_conv_filters=[[32, 64], [64, 128], [256, 256]],
        stackwise_pooling=[True, True, False],
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_conv_filters,
        stackwise_pooling,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if len(stackwise_conv_filters) != len(stackwise_pooling):
            raise ValueError("All stackwise args should have the same length.")

        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        num_blocks = len(stackwise_conv_filters)

        # Layer shorcuts with common args.
        norm = functools.partial(
            layers.BatchNormalization,
            axis=channel_axis,
            dtype=dtype,
        )
        act = functools.partial(
            layers.Activation,
            activation="relu",
            dtype=dtype,
        )
        conv = functools.partial(
            layers.Conv2D,
            kernel_size=(3, 3),
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
        )
        sep_conv = functools.partial(
            layers.SeparableConv2D,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
        )
        point_conv = functools.partial(
            layers.Conv2D,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
        )
        pool = functools.partial(
            layers.MaxPool2D,
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same",
            data_format=data_format,
            dtype=dtype,
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape)
        x = image_input  # Intermediate result.

        # Iterate through the blocks.
        for block_i in range(num_blocks):
            first_block, last_block = block_i == 0, block_i == num_blocks - 1
            block_filters = stackwise_conv_filters[block_i]
            use_pooling = stackwise_pooling[block_i]

            # Save the block input as a residual.
            residual = x
            for conv_i, filters in enumerate(block_filters):
                # First block has post activation and strides on first conv.
                if first_block:
                    prefix = f"block{block_i + 1}_conv{conv_i + 1}"
                    strides = (2, 2) if conv_i == 0 else (1, 1)
                    x = conv(filters, strides=strides, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)
                    x = act(name=f"{prefix}_act")(x)
                # Last block has post activation.
                elif last_block:
                    prefix = f"block{block_i + 1}_sepconv{conv_i + 1}"
                    x = sep_conv(filters, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)
                    x = act(name=f"{prefix}_act")(x)
                else:
                    prefix = f"block{block_i + 1}_sepconv{conv_i + 1}"
                    # The first conv in second block has no activation.
                    if block_i != 1 or conv_i != 0:
                        x = act(name=f"{prefix}_act")(x)
                    x = sep_conv(filters, name=prefix)(x)
                    x = norm(name=f"{prefix}_bn")(x)

            # Optional block pooling.
            if use_pooling:
                x = pool(name=f"block{block_i + 1}_pool")(x)

            # Sum residual, first and last block do not have a residual.
            if not first_block and not last_block:
                prefix = f"block{block_i + 1}_residual"
                filters = x.shape[channel_axis]
                # Match filters with a pointwise conv if needed.
                if filters != residual.shape[channel_axis]:
                    residual = point_conv(filters, name=f"{prefix}_conv")(
                        residual
                    )
                    residual = norm(name=f"{prefix}_bn")(residual)
                x = layers.Add(name=f"{prefix}_add", dtype=dtype)([x, residual])

        super().__init__(
            inputs=image_input,
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.stackwise_conv_filters = stackwise_conv_filters
        self.stackwise_pooling = stackwise_pooling
        self.image_shape = image_shape
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_conv_filters": self.stackwise_conv_filters,
                "stackwise_pooling": self.stackwise_pooling,
                "image_shape": self.image_shape,
            }
        )
        return config
