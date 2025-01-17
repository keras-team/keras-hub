import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobilenet.conv_bn_act_block import ConvBnActBlock
from keras_hub.src.models.mobilenet.depthwise_conv_block import (
    DepthwiseConvBlock,
)
from keras_hub.src.models.mobilenet.inverted_residual_block import (
    InvertedResidualBlock,
)
from keras_hub.src.models.mobilenet.util import adjust_channels

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9


@keras_hub_export("keras_hub.models.MobileNetBackbone")
class MobileNetBackbone(Backbone):
    """Instantiates the MobileNet architecture.

    MobileNet is a lightweight convolutional neural network (CNN)
    optimized for mobile and edge devices, striking a balance between
    accuracy and efficiency. By employing depthwise separable convolutions
    and techniques like Squeeze-and-Excitation (SE) blocks,
    MobileNet models are highly suitable for real-time applications on
    resource-constrained devices.

    References:
        - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
        (ICCV 2019)

    Args:
        stackwise_expansion: list of list of ints, the expanded filters for
            each inverted residual block for each block in the model.
        stackwise_num_blocks: list of ints, number of inversted residual blocks
            per block
        stackwise_num_filters: list of list of ints, number of filters for
            each inverted residual block in the model.
        stackwise_kernel_size: list of list of ints, kernel size for each
            inverted residual block in the model.
        stackwise_num_strides: list of list of ints, stride length for each
            inverted residual block in the model.
        stackwise_se_ratio: se ratio for each inverted residual block in the
            model. 0 if dont want to add Squeeze and Excite layer.
        stackwise_activation: list of list of activation functions, for each
            inverted residual block in the model.
        image_shape: optional shape tuple, defaults to (224, 224, 3).
        depth_multiplier: float, controls the width of the network.
            - If `depth_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `depth_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `depth_multiplier` = 1, default number of filters from the
                paper are used at each layer.
        input_num_filters: number of filters in first convolution layer
        output_num_filters: specifies whether to add conv and batch_norm in the
            end, if set to None, it will not add these layers in the end.
            'None' for MobileNetV1
        input_activation: activation function to be used in the input layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        output_activation: activation function to be used in the output layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        depthwise_filters: int, number of filters in depthwise separable
            convolution layer
        squeeze_and_excite: float, squeeze and excite ratio in the depthwise
            layer, None, if dont want to do squeeze and excite


    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetBackbone(
        stackwise_expansion=[
                [40, 56],
                [64, 144, 144],
                [72, 72],
                [144, 288, 288],
            ],
            stackwise_num_blocks=[2, 3, 2, 3],
            stackwise_num_filters=[
                [16, 16],
                [24, 24, 24],
                [24, 24],
                [48, 48, 48],
            ],
            stackwise_kernel_size=[[3, 3], [5, 5, 5], [5, 5], [5, 5, 5]],
            stackwise_num_strides=[[2, 1], [2, 1, 1], [1, 1], [2, 1, 1]],
            stackwise_se_ratio=[
                [None, None],
                [0.25, 0.25, 0.25],
                [0.3, 0.3],
                [0.3, 0.25, 0.25],
            ],
            stackwise_activation=[
                ["relu", "relu"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish", "hard_swish"],
            ],
            output_num_filters=288,
            input_activation="hard_swish",
            output_activation="hard_swish",
            input_num_filters=16,
            image_shape=(224, 224, 3),
            depthwise_filters=8,
            squeeze_and_excite=0.5,

    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_expansion,
        stackwise_num_blocks,
        stackwise_num_filters,
        stackwise_kernel_size,
        stackwise_num_strides,
        stackwise_se_ratio,
        stackwise_activation,
        stackwise_padding,
        output_num_filters,
        depthwise_filters,
        last_layer_filter,
        squeeze_and_excite=None,
        image_shape=(None, None, 3),
        input_activation="hard_swish",
        output_activation="hard_swish",
        input_num_filters=16,
        **kwargs,
    ):
        # === Functional Model ===
        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )

        image_input = keras.layers.Input(shape=image_shape)
        x = image_input
        input_num_filters = adjust_channels(input_num_filters)

        x = keras.layers.ZeroPadding2D(
            padding=(1, 1),
            name="input_pad",
        )(x)
        x = keras.layers.Conv2D(
            input_num_filters,
            kernel_size=3,
            strides=(2, 2),
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name="input_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="input_batch_norm",
        )(x)
        x = keras.layers.Activation(input_activation)(x)

        x = DepthwiseConvBlock(
            input_num_filters,
            depthwise_filters,
            se=squeeze_and_excite,
            name="block_0",
        )(x)

        for block in range(len(stackwise_num_blocks)):
            for inverted_block in range(stackwise_num_blocks[block]):
                infilters = x.shape[channel_axis]
                x = InvertedResidualBlock(
                    expansion=stackwise_expansion[block][inverted_block],
                    infilters=infilters,
                    filters=adjust_channels(
                        stackwise_num_filters[block][inverted_block]
                    ),
                    kernel_size=stackwise_kernel_size[block][inverted_block],
                    stride=stackwise_num_strides[block][inverted_block],
                    se_ratio=stackwise_se_ratio[block][inverted_block],
                    activation=stackwise_activation[block][inverted_block],
                    padding=stackwise_padding[block][inverted_block],
                    name=f"block_{block+1}_{inverted_block}",
                )(x)

        x = ConvBnActBlock(
            filter=adjust_channels(last_layer_filter),
            activation="hard_swish",
            name=f"block_{len(stackwise_num_blocks)+1}_0",
        )(x)

        super().__init__(inputs=image_input, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_expansion = stackwise_expansion
        self.stackwise_num_blocks = stackwise_num_blocks
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_num_strides = stackwise_num_strides
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.stackwise_padding = stackwise_padding
        self.input_num_filters = input_num_filters
        self.output_num_filters = output_num_filters
        self.depthwise_filters = depthwise_filters
        self.last_layer_filter = last_layer_filter
        self.squeeze_and_excite = squeeze_and_excite
        self.input_activation = keras.activations.get(input_activation)
        self.output_activation = keras.activations.get(output_activation)
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_num_strides": self.stackwise_num_strides,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "stackwise_padding": self.stackwise_padding,
                "image_shape": self.image_shape,
                "input_num_filters": self.input_num_filters,
                "output_num_filters": self.output_num_filters,
                "depthwise_filters": self.depthwise_filters,
                "last_layer_filter": self.last_layer_filter,
                "squeeze_and_excite": self.squeeze_and_excite,
                "input_activation": keras.activations.serialize(
                    activation=self.input_activation
                ),
                "output_activation": keras.activations.serialize(
                    activation=self.output_activation
                ),
            }
        )
        return config
