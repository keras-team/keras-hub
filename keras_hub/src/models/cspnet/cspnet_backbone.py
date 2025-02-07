import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.CSPNetBackbone")
class CSPNetBackbone(FeaturePyramidBackbone):
    """This class represents Keras Backbone of CSPNet model.

    This class implements a CSPNet backbone as described in
    [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](
        https://arxiv.org/abs/1911.11929).

    Args:
        stem_filters: int or list of ints, filter size for the stem.
        stem_kernel_size: int or tuple/list of 2 integers, kernel size for the
            stem.
        stem_strides: int or tuple/list of 2 integers, stride length of the
            convolution for the stem.
        stackwise_num_filters: A list of ints, filter size for each block level
            in the model.
        stackwise_strides: int or tuple/list of ints, strides for each block
            level in the model.
        stackwise_depth: A list of ints, representing the depth
            (number of blocks) for each block level in the model.
        block_type: str. One of `"bottleneck_block"`, `"dark_block"`, or
            `"edge_block"`. Use `"dark_block"` for DarkNet blocks,
            `"edge_block"` for EdgeResidual / Fused-MBConv blocks.
        groups: int, specifying the number of groups into which the input is
            split along the channel axis. Defaults to `1`.
        stage_type: str. One of `"csp"`, `"dark"`, or `"cs3"`. Use `"dark"` for
            DarkNet stages, `"csp"` for Cross Stage, and `"cs3"` for Cross Stage
            with only one transition conv. Defaults to `None`, which defaults to
            `"cs3"`.
        activation: str. Activation function for the model.
        output_strides: int, output stride length of the backbone model. Must be
            one of `(8, 16, 32)`. Defaults to `32`.
        bottle_ratio: float or tuple/list of floats. The dimensionality of the
            intermediate bottleneck space (i.e., the number of output filters in
            the bottleneck convolution), calculated as
            `(filters * bottle_ratio)` and applied to:
            - the first convolution of `"dark_block"` and `"edge_block"`
            - the first two convolutions of `"bottleneck_block"`
            of each stage. Defaults to `1.0`.
        block_ratio: float or tuple/list of floats. Filter size for each block,
            calculated as `(stackwise_num_filters * block_ratio)` for each
            stage. Defaults to `1.0`.
        expand_ratio: float or tuple/list of floats. Filters ratio for `"csp"`
            and `"cs3"` stages at different levels. Defaults to `1.0`.
        stem_padding: str, padding value for the stem, either `"valid"` or
            `"same"`. Defaults to `"valid"`.
        stem_pooling: str, pooling value for the stem. Defaults to `None`.
        avg_down: bool, if `True`, `AveragePooling2D` is applied at the
            beginning of each stage when `strides == 2`. Defaults to `False`.
        down_growth: bool, grow downsample channels to output channels. Applies
            to Cross Stage only. Defaults to `False`.
        cross_linear: bool, if `True`, activation will not be applied after the
            expansion convolution. Applies to Cross Stage only. Defaults to
            `False`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_hub.models.CSPNetBackbone.from_preset(
        "cspdarknet53_ra_imagenet"
    )
    model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_hub.models.CSPNetBackbone(
        stem_filters=32,
        stem_kernel_size=3,
        stem_strides=1,
        stackwise_depth=[1, 2, 4],
        stackwise_strides=[1, 2, 2],
        stackwise_num_filters=[32, 64, 128],
        block_type="dark,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stem_filters,
        stem_kernel_size,
        stem_strides,
        stackwise_depth,
        stackwise_strides,
        stackwise_num_filters,
        block_type,
        groups=1,
        stage_type=None,
        activation="leaky_relu",
        output_strides=32,
        bottle_ratio=[1.0],
        block_ratio=[1.0],
        expand_ratio=[1.0],
        stem_padding="valid",
        stem_pooling=None,
        avg_down=False,
        down_growth=False,
        cross_linear=False,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if block_type not in (
            "dark_block",
            "edge_block",
            "bottleneck_block",
        ):
            raise ValueError(
                '`block_type` must be either `"dark_block"`, '
                '`"edge_block"`, or `"bottleneck_block"`.'
                f"Received block_type={block_type}."
            )

        if stage_type not in (
            "dark",
            "csp",
            "cs3",
        ):
            raise ValueError(
                '`block_type` must be either `"dark"`, `"csp"`, or `"cs3"`.'
                f"Received block_type={stage_type}."
            )
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape)
        x = image_input  # Intermediate result.
        stem, stem_feat_info = create_csp_stem(
            data_format=data_format,
            channel_axis=channel_axis,
            filters=stem_filters,
            kernel_size=stem_kernel_size,
            strides=stem_strides,
            pooling=stem_pooling,
            padding=stem_padding,
            activation=activation,
            dtype=dtype,
        )(x)

        stages, pyramid_outputs = create_csp_stages(
            inputs=stem,
            filters=stackwise_num_filters,
            data_format=data_format,
            channel_axis=channel_axis,
            stackwise_depth=stackwise_depth,
            reduction=stem_feat_info,
            groups=groups,
            block_ratio=block_ratio,
            bottle_ratio=bottle_ratio,
            expand_ratio=expand_ratio,
            strides=stackwise_strides,
            avg_down=avg_down,
            down_growth=down_growth,
            cross_linear=cross_linear,
            activation=activation,
            output_strides=output_strides,
            stage_type=stage_type,
            block_type=block_type,
            dtype=dtype,
            name="csp_stage",
        )

        super().__init__(
            inputs=image_input, outputs=stages, dtype=dtype, **kwargs
        )

        # === Config ===
        self.stem_filters = stem_filters
        self.stem_kernel_size = stem_filters
        self.stem_strides = stem_strides
        self.stackwise_depth = stackwise_depth
        self.stackwise_strides = stackwise_strides
        self.stackwise_num_filters = stackwise_num_filters
        self.stage_type = stage_type
        self.block_type = block_type
        self.output_strides = output_strides
        self.groups = groups
        self.activation = activation
        self.bottle_ratio = bottle_ratio
        self.block_ratio = block_ratio
        self.expand_ratio = expand_ratio
        self.stem_padding = stem_padding
        self.stem_pooling = stem_pooling
        self.avg_down = avg_down
        self.down_growth = down_growth
        self.cross_linear = cross_linear
        self.image_shape = image_shape
        self.data_format = data_format
        self.image_shape = image_shape
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_filters": self.stem_filters,
                "stem_kernel_size": self.stem_kernel_size,
                "stem_strides": self.stem_strides,
                "stackwise_depth": self.stackwise_depth,
                "stackwise_strides": self.stackwise_strides,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stage_type": self.stage_type,
                "block_type": self.block_type,
                "output_strides": self.output_strides,
                "groups": self.groups,
                "activation": self.activation,
                "bottle_ratio": self.bottle_ratio,
                "block_ratio": self.block_ratio,
                "expand_ratio": self.expand_ratio,
                "stem_padding": self.stem_padding,
                "stem_pooling": self.stem_pooling,
                "avg_down": self.avg_down,
                "down_growth": self.down_growth,
                "cross_linear": self.cross_linear,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
                "pyramid_outputs": self.pyramid_outputs,
            }
        )
        return config


def bottleneck_block(
    filters,
    channel_axis,
    data_format,
    bottle_ratio,
    dilation=1,
    groups=1,
    activation="relu",
    dtype=None,
    name=None,
):
    """
    BottleNeck block.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
            (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        bottle_ratio: float, ratio for bottleneck filters. Number of bottleneck
            `filters = filters * bottle_ratio`.
        dilation: int or tuple/list of 2 integers, specifying the dilation rate
            to use for dilated convolution, defaults to `1`.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis
        activation: Activation for the conv layers, defaults to "relu".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        Output tensor of block.
    """
    if name is None:
        name = f"bottleneck{keras.backend.get_uid('bottleneck')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_block_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_block_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=3,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_block_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_block_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_block_conv_3",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_block_bn_3",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_3",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_3",
            )(x)

        x = layers.add([x, shortcut], name=f"{name}_bottleneck_block_add")
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_4",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_4",
            )(x)
        return x

    return apply


def dark_block(
    filters,
    data_format,
    channel_axis,
    dilation,
    bottle_ratio,
    groups,
    activation,
    dtype=None,
    name=None,
):
    """
    DarkNet block.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
            (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        bottle_ratio: float, ratio for darknet filters. Number of darknet
            `filters = filters * bottle_ratio`.
        dilation: int or tuple/list of 2 integers, specifying the dilation rate
            to use for dilated convolution, defaults to `1`.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis
        activation: Activation for the conv layers, defaults to "relu".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        Output tensor of block.
    """
    if name is None:
        name = f"dark{keras.backend.get_uid('dark')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_dark_block_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_dark_block_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_dark_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_dark_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_dark_block_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_dark_block_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_dark_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_dark_block_activation_2",
            )(x)

        x = layers.add([x, shortcut], name=f"{name}_dark_block_add")
        return x

    return apply


def edge_block(
    filters,
    data_format,
    channel_axis,
    dilation=1,
    bottle_ratio=0.5,
    groups=1,
    activation="relu",
    dtype=None,
    name=None,
):
    """
    EdgeResidual / Fused-MBConv blocks.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
            (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        bottle_ratio: float, ratio for edge_block filters. Number of edge_block
            `filters = filters * bottle_ratio`.
        dilation: int or tuple/list of 2 integers, specifying the dilation rate
            to use for dilated convolution, defaults to `1`.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis
        activation: Activation for the conv layers, defaults to "relu".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        Output tensor of block.
    """
    if name is None:
        name = f"edge{keras.backend.get_uid('edge')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=3,
            use_bias=False,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_edge_block_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_edge_block_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_edge_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_edge_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_edge_block_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_edge_block_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_edge_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_edge_block_activation_2",
            )(x)

        x = layers.add([x, shortcut], name=f"{name}_edge_block_add")
        return x

    return apply


def cross_stage(
    filters,
    strides,
    dilation,
    depth,
    data_format,
    channel_axis,
    block_ratio=1.0,
    bottle_ratio=1.0,
    expand_ratio=1.0,
    groups=1,
    first_dilation=None,
    avg_down=False,
    activation="relu",
    down_growth=False,
    cross_linear=False,
    block_fn=bottleneck_block,
    dtype=None,
    name=None,
):
    """ "
    Cross Stage.
    """
    if name is None:
        name = f"cross_stage_{keras.backend.get_uid('cross_stage')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        prev_filters = keras.ops.shape(x)[channel_axis]
        down_chs = filters if down_growth else prev_filters
        expand_chs = int(round(filters * expand_ratio))
        block_channels = int(round(filters * block_ratio))

        if strides != 1 or first_dilation != dilation:
            if avg_down:
                if strides == 2:
                    x = layers.AveragePooling2D(
                        2, dtype=dtype, name=f"{name}_csp_avg_pool"
                    )(x)
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_csp_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_csp_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.01,
                        dtype=dtype,
                        name=f"{name}_csp_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_csp_activation_1",
                    )(x)
            else:
                x = layers.Conv2D(
                    filters=down_chs,
                    kernel_size=3,
                    strides=strides,
                    dilation_rate=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_csp_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_csp_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.01,
                        dtype=dtype,
                        name=f"{name}_csp_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_csp_activation_1",
                    )(x)

        x = layers.Conv2D(
            filters=expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_csp_conv_exp",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_csp_bn_2",
        )(x)
        if not cross_linear:
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.01,
                    dtype=dtype,
                    name=f"{name}_csp_activation_2",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_csp_activation_2",
                )(x)
        prev_filters = keras.ops.shape(x)[channel_axis]
        xs, xb = ops.split(
            x,
            indices_or_sections=prev_filters // (expand_chs // 2),
            axis=channel_axis,
        )

        for i in range(depth):
            xb = block_fn(
                filters=block_channels,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                activation=activation,
                data_format=data_format,
                channel_axis=channel_axis,
                dtype=dtype,
                name=f"{name}_block_{i}",
            )(xb)

        xb = layers.Conv2D(
            filters=expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_csp_conv_transition_b",
        )(xb)
        xb = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_csp_transition_b_bn",
        )(xb)
        if activation == "leaky_relu":
            xb = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_csp_transition_b_activation",
            )(xb)
        else:
            xb = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_csp_transition_b_activation",
            )(xb)

        out = layers.Concatenate(
            axis=channel_axis, name=f"{name}_csp_conv_concat"
        )([xs, xb])
        out = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_csp_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_csp_transition_bn",
        )(out)
        if activation == "leaky_relu":
            out = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_csp_transition_activation",
            )(out)
        else:
            out = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_csp_transition_activation",
            )(out)
        return out

    return apply


def cross_stage3(
    data_format,
    channel_axis,
    filters,
    strides,
    dilation,
    depth,
    block_ratio,
    bottle_ratio,
    expand_ratio,
    avg_down,
    activation,
    first_dilation,
    down_growth,
    cross_linear,
    block_fn,
    groups,
    name=None,
    dtype=None,
):
    """
    Cross Stage 3.

    Similar to Cross Stage, but with only one transition conv in the output.
    """
    if name is None:
        name = f"cross_stage3_{keras.backend.get_uid('cross_stage3')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        prev_filters = keras.ops.shape(x)[channel_axis]
        down_chs = filters if down_growth else prev_filters
        expand_chs = int(round(filters * expand_ratio))
        block_filters = int(round(filters * block_ratio))

        if strides != 1 or first_dilation != dilation:
            if avg_down:
                if strides == 2:
                    x = layers.AveragePooling2D(
                        2, dtype=dtype, name=f"{name}_cross_stage3_avg_pool"
                    )(x)
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cs3_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_cs3_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.01,
                        dtype=dtype,
                        name=f"{name}_cs3_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cs3_activation_1",
                    )(x)
            else:
                x = layers.Conv2D(
                    filters=down_chs,
                    kernel_size=3,
                    strides=strides,
                    dilation_rate=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cs3_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_cs3_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.01,
                        dtype=dtype,
                        name=f"{name}_cs3__activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cs3_activation_1",
                    )(x)

        x = layers.Conv2D(
            filters=expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_cs3_conv_exp",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_cs3_bn_2",
        )(x)
        if not cross_linear:
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.01,
                    dtype=dtype,
                    name=f"{name}_cs3_activation_2",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_cs3_activation_2",
                )(x)

        prev_filters = keras.ops.shape(x)[channel_axis]
        x1, x2 = ops.split(
            x,
            indices_or_sections=prev_filters // (expand_chs // 2),
            axis=channel_axis,
        )

        for i in range(depth):
            x1 = block_fn(
                filters=block_filters,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                activation=activation,
                data_format=data_format,
                channel_axis=channel_axis,
                dtype=dtype,
                name=f"{name}_block_{i}",
            )(x1)

        out = layers.Concatenate(
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_cs3_conv_transition_concat",
        )([x1, x2])
        out = layers.Conv2D(
            filters=expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_cs3_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_cs3_transition_bn",
        )(out)
        if activation == "leaky_relu":
            out = layers.LeakyReLU(
                negative_slope=0.01,
                dtype=dtype,
                name=f"{name}_cs3_activation_3",
            )(out)
        else:
            out = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_cs3_activation_3",
            )(out)
        return out

    return apply


def dark_stage(
    data_format,
    channel_axis,
    filters,
    strides,
    dilation,
    depth,
    block_ratio,
    bottle_ratio,
    avg_down,
    activation,
    first_dilation,
    block_fn,
    groups,
    expand_ratio=None,
    down_growth=None,
    cross_linear=None,
    name=None,
    dtype=None,
):
    """
    DarkNet Stage.

    Similar to DarkNet Stage, but with only one transition conv in the output.
    """
    if name is None:
        name = f"dark_stage_{keras.backend.get_uid('dark_stage')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        block_channels = int(round(filters * block_ratio))
        if avg_down:
            if strides == 2:
                x = layers.AveragePooling2D(
                    2, dtype=dtype, name=f"{name}_dark_avg_pool"
                )(x)
            x = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                use_bias=False,
                groups=groups,
                data_format=data_format,
                dtype=dtype,
                name=f"{name}_dark_conv_down_1",
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-05,
                axis=channel_axis,
                dtype=dtype,
                name=f"{name}_dark_bn_1",
            )(x)
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.01,
                    dtype=dtype,
                    name=f"{name}_dark_activation_1",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_dark_activation_1",
                )(x)
        else:
            x = layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=strides,
                dilation_rate=first_dilation,
                use_bias=False,
                groups=groups,
                data_format=data_format,
                dtype=dtype,
                name=f"{name}_dark_conv_down_1",
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-05,
                axis=channel_axis,
                dtype=dtype,
                name=f"{name}_dark_bn_1",
            )(x)
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.01,
                    dtype=dtype,
                    name=f"{name}_dark_activation_1",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_dark_activation_1",
                )(x)
            for i in range(depth):
                x = block_fn(
                    filters=block_channels,
                    dilation=dilation,
                    bottle_ratio=bottle_ratio,
                    groups=groups,
                    activation=activation,
                    data_format=data_format,
                    channel_axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_block_{i}",
                )(x)
        return x

    return apply


def create_csp_stem(
    data_format,
    channel_axis,
    activation,
    padding,
    filters=32,
    kernel_size=3,
    strides=2,
    pooling=None,
    dtype=None,
):
    if not isinstance(filters, (tuple, list)):
        filters = [filters]
    stem_depth = len(filters)
    assert stem_depth
    assert strides in (1, 2, 4)
    last_idx = stem_depth - 1

    def apply(x):
        stem_strides = 1
        for i, chs in enumerate(filters):
            conv_strides = (
                2
                if (i == 0 and strides > 1)
                or (i == last_idx and strides > 2 and not pooling)
                else 1
            )
            x = layers.Conv2D(
                filters=chs,
                kernel_size=kernel_size,
                strides=conv_strides,
                padding=padding if i == 0 else "valid",
                use_bias=False,
                data_format=data_format,
                dtype=dtype,
                name=f"csp_stem_conv_{i}",
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-05,
                axis=channel_axis,
                dtype=dtype,
                name=f"csp_stem_bn_{i}",
            )(x)
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.01,
                    dtype=dtype,
                    name=f"csp_stem_activation_{i}",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"csp_stem_activation_{i}",
                )(x)
            stem_strides *= conv_strides

        if pooling == "max":
            assert strides > 2
            x = layers.MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="same",
                data_format=data_format,
                dtype=dtype,
                name="csp_stem_pool",
            )(x)
            stem_strides *= 2
        return x, stem_strides

    return apply


def create_csp_stages(
    inputs,
    filters,
    data_format,
    channel_axis,
    stackwise_depth,
    reduction,
    block_ratio,
    bottle_ratio,
    expand_ratio,
    strides,
    groups,
    avg_down,
    down_growth,
    cross_linear,
    activation,
    output_strides,
    stage_type,
    block_type,
    dtype,
    name,
):
    if name is None:
        name = f"csp_stage_{keras.backend.get_uid('csp_stage')}"

    num_stages = len(stackwise_depth)
    dilation = 1
    net_strides = reduction
    strides = _pad_arg(strides, num_stages)
    expand_ratio = _pad_arg(expand_ratio, num_stages)
    bottle_ratio = _pad_arg(bottle_ratio, num_stages)
    block_ratio = _pad_arg(block_ratio, num_stages)

    if stage_type == "dark":
        stage_fn = dark_stage
    elif stage_type == "csp":
        stage_fn = cross_stage
    else:
        stage_fn = cross_stage3

    if block_type == "dark_block":
        block_fn = dark_block
    elif block_type == "edge_block":
        block_fn = edge_block
    else:
        block_fn = bottleneck_block

    stages = inputs
    pyramid_outputs = {}
    for stage_idx, _ in enumerate(stackwise_depth):
        if net_strides >= output_strides and strides[stage_idx] > 1:
            dilation *= strides[stage_idx]
            strides = 1
        net_strides *= strides[stage_idx]
        first_dilation = 1 if dilation in (1, 2) else 2
        stages = stage_fn(
            data_format=data_format,
            channel_axis=channel_axis,
            filters=filters[stage_idx],
            depth=stackwise_depth[stage_idx],
            strides=strides[stage_idx],
            dilation=dilation,
            block_ratio=block_ratio[stage_idx],
            bottle_ratio=bottle_ratio[stage_idx],
            expand_ratio=expand_ratio[stage_idx],
            groups=groups,
            first_dilation=first_dilation,
            avg_down=avg_down,
            activation=activation,
            down_growth=down_growth,
            cross_linear=cross_linear,
            block_fn=block_fn,
            dtype=dtype,
            name=f"stage_{stage_idx}",
        )(stages)
        pyramid_outputs[f"P{stage_idx + 2}"] = stages
    return stages, pyramid_outputs


def _pad_arg(x, n):
    """
    pads an argument tuple to specified n by padding with last value
    """
    if not isinstance(x, (tuple, list)):
        x = (x,)
    curr_n = len(x)
    pad_n = n - curr_n
    if pad_n <= 0:
        return x[:n]
    return tuple(x + (x[-1],) * pad_n)
