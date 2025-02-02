from keras import ops
from keras.backend import is_keras_tensor
from keras.layers import Input
from keras.layers import MaxPooling2D

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_conv_bn
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_CSP


def build_input_tensor(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        input_tensor = Input(shape=input_shape, **kwargs)
    else:
        if not is_keras_tensor(input_tensor):
            input_tensor = Input(input_shape, tensor=input_tensor, **kwargs)
    return input_tensor


def apply_stem(x, stem_width, activation):
    x = apply_conv_bn(x, stem_width // 2, 3, 2, activation, "stem_1")
    x = apply_conv_bn(x, stem_width, 3, 2, activation, "stem_2")
    return x


def apply_fast_SPP(x, pool_size=5, activation="swish", name="spp_fast"):
    input_channels = x.shape[-1]
    hidden_channels = int(input_channels // 2)
    x = apply_conv_bn(x, hidden_channels, 1, 1, activation, f"{name}_pre")
    pool_kwargs = {"strides": 1, "padding": "same"}
    p1 = MaxPooling2D(pool_size, **pool_kwargs, name=f"{name}_pool1")(x)
    p2 = MaxPooling2D(pool_size, **pool_kwargs, name=f"{name}_pool2")(p1)
    p3 = MaxPooling2D(pool_size, **pool_kwargs, name=f"{name}_pool3")(p2)
    x = ops.concatenate([x, p1, p2, p3], axis=-1)
    x = apply_conv_bn(x, input_channels, 1, 1, activation, f"{name}_output")
    return x


def apply_yolo_block(x, block_arg, channels, depth, block_depth, activation):
    name = f"stack{block_arg + 1}"
    if block_arg >= 1:
        x = apply_conv_bn(x, channels, 3, 2, activation, f"{name}_downsample")
    x = apply_CSP(x, -1, depth, True, 0.5, activation, f"{name}_c2f")
    if block_arg == len(block_depth) - 1:
        x = apply_fast_SPP(x, 5, activation, f"{name}_spp_fast")
    return x


def stackwise_yolo_blocks(x, stackwise_depth, stackwise_channels, activation):
    pyramid_level_inputs = {"P1": get_tensor_input_name(x)}
    iterator = enumerate(zip(stackwise_channels, stackwise_depth))
    block_args = (stackwise_depth, activation)
    for stack_arg, (channel, depth) in iterator:
        x = apply_yolo_block(x, stack_arg, channel, depth, *block_args)
        pyramid_level_inputs[f"P{stack_arg + 2}"] = get_tensor_input_name(x)
    return x, pyramid_level_inputs


def remove_batch_dimension(input_shape):
    return input_shape[1:]


def get_tensor_input_name(tensor):
    return tensor._keras_history.operation.name


@keras_hub_export("keras_hub.models.YOLOV8Backbone")
class YOLOV8Backbone(Backbone):
    """Implements the YOLOV8 backbone for object detection.

    This backbone is a variant of the `CSPDarkNetBackbone` architecture.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/
    transfer_learning/).

    Args:
        stackwise_channels: A list of int. The number of channels for each dark
            level in the model.
        stackwise_depth: A list of int. The depth for each dark level in the
            model.
        include_rescaling: bool. Rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        activation: str. The activation functions to use in the backbone to
            use in the CSPDarkNet blocks. Defaults to "swish".
        input_shape: optional shape tuple, defaults to `(None, None, 3)`.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Returns:
        A `keras.Model` instance.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_hub.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_hub.models.YOLOV8Backbone(
        stackwise_channels=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_channels,
        stackwise_depth,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = build_input_tensor(input_shape, input_tensor)
        stem_width = stackwise_channels[0]
        x = apply_stem(inputs, stem_width, activation)
        x, pyramid_level_inputs = stackwise_yolo_blocks(
            x, stackwise_depth, stackwise_channels, activation
        )
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": remove_batch_dimension(self.input_shape),
                "stackwise_channels": self.stackwise_channels,
                "stackwise_depth": self.stackwise_depth,
                "activation": self.activation,
            }
        )
        return config
