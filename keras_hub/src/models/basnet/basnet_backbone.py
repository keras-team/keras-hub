import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.resnet.resnet_backbone import (
    apply_basic_block as resnet_basic_block,
)


@keras_hub_export("keras_hub.models.BASNetBackbone")
class BASNetBackbone(Backbone):
    """BASNet architecture for semantic segmentation.

    A Keras model implementing the BASNet architecture described in [BASNet:
    Boundary-Aware Segmentation Network for Mobile and Web Applications](
    https://arxiv.org/abs/2101.04704). BASNet uses a predict-refine
    architecture for highly accurate image segmentation.

    Args:
        image_encoder: A `keras_hub.models.ResNetBackbone` instance. The
            backbone network for the model that is used as a feature extractor
            for BASNet prediction encoder. Currently supported backbones are
            ResNet18 and ResNet34.
            (Note: Do not specify `image_shape` within the backbone.
            Please provide these while initializing the 'BASNetBackbone' model)
        num_classes: int, the number of classes for the segmentation model.
        image_shape: optional shape tuple, defaults to (None, None, 3).
        projection_filters: int, number of filters in the convolution layer
            projecting low-level features from the `backbone`.
        prediction_heads: (Optional) List of `keras.layers.Layer` defining
            the prediction module head for the model. If not provided, a
            default head is created with a Conv2D layer followed by resizing.
        refinement_head: (Optional) a `keras.layers.Layer` defining the
            refinement module head for the model. If not provided, a default
            head is created with a Conv2D layer.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.
    """

    def __init__(
        self,
        image_encoder,
        num_classes,
        image_shape=(None, None, 3),
        projection_filters=64,
        prediction_heads=None,
        refinement_head=None,
        dtype=None,
        **kwargs,
    ):
        if not isinstance(image_encoder, keras.layers.Layer) or not isinstance(
            image_encoder, keras.Model
        ):
            raise ValueError(
                "Argument `image_encoder` must be a `keras.layers.Layer`"
                f" instance or `keras.Model`. Received instead"
                f" image_encoder={image_encoder} (of type"
                f" {type(image_encoder)})."
            )

        if tuple(image_encoder.image_shape) != (None, None, 3):
            raise ValueError(
                "Do not specify `image_shape` within the"
                " `BASNetBackbone`'s image_encoder. \nPlease provide"
                " `image_shape` while initializing the 'BASNetBackbone' model."
            )

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)
        x = inputs

        if prediction_heads is None:
            prediction_heads = []
            for size in (1, 2, 4, 8, 16, 32, 32):
                head_layers = [
                    keras.layers.Conv2D(
                        num_classes,
                        kernel_size=(3, 3),
                        padding="same",
                        dtype=dtype,
                    )
                ]
                if size != 1:
                    head_layers.append(
                        keras.layers.UpSampling2D(
                            size=size, interpolation="bilinear", dtype=dtype
                        )
                    )
                prediction_heads.append(keras.Sequential(head_layers))

        if refinement_head is None:
            refinement_head = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        num_classes,
                        kernel_size=(3, 3),
                        padding="same",
                        dtype=dtype,
                    ),
                ]
            )

        # Prediction model.
        predict_model = basnet_predict(
            x, image_encoder, projection_filters, prediction_heads, dtype=dtype
        )

        # Refinement model.
        refine_model = basnet_rrm(
            predict_model, projection_filters, refinement_head, dtype=dtype
        )

        outputs = refine_model.outputs  # Combine outputs.
        outputs.extend(predict_model.outputs)

        output_names = ["refine_out"] + [
            f"predict_out_{i}" for i in range(1, len(outputs))
        ]

        outputs = {
            output_name: keras.layers.Activation(
                "sigmoid", name=output_name, dtype=dtype
            )(output)
            for output, output_name in zip(outputs, output_names)
        }

        super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)

        # === Config ===
        self.image_encoder = image_encoder
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.projection_filters = projection_filters
        self.prediction_heads = prediction_heads
        self.refinement_head = refinement_head

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.saving.serialize_keras_object(
                    self.image_encoder
                ),
                "num_classes": self.num_classes,
                "image_shape": self.image_shape,
                "projection_filters": self.projection_filters,
                "prediction_heads": [
                    keras.saving.serialize_keras_object(prediction_head)
                    for prediction_head in self.prediction_heads
                ],
                "refinement_head": keras.saving.serialize_keras_object(
                    self.refinement_head
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config:
            config["image_encoder"] = keras.layers.deserialize(
                config["image_encoder"]
            )
        if "prediction_heads" in config and isinstance(
            config["prediction_heads"], list
        ):
            for i in range(len(config["prediction_heads"])):
                if isinstance(config["prediction_heads"][i], dict):
                    config["prediction_heads"][i] = keras.layers.deserialize(
                        config["prediction_heads"][i]
                    )

        if "refinement_head" in config and isinstance(
            config["refinement_head"], dict
        ):
            config["refinement_head"] = keras.layers.deserialize(
                config["refinement_head"]
            )
        return super().from_config(config)


def convolution_block(x_input, filters, dilation=1, dtype=None):
    """Apply convolution + batch normalization + ReLU activation.

    Args:
        x_input: Input keras tensor.
        filters: int, number of output filters in the convolution.
        dilation: int, dilation rate for the convolution operation.
            Defaults to 1.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Returns:
        A tensor with convolution, batch normalization, and ReLU
        activation applied.
    """
    x = keras.layers.Conv2D(
        filters, (3, 3), padding="same", dilation_rate=dilation, dtype=dtype
    )(x_input)
    x = keras.layers.BatchNormalization(dtype=dtype)(x)
    return keras.layers.Activation("relu", dtype=dtype)(x)


def get_resnet_block(_resnet, block_num):
    """Extract and return a specific ResNet block.

    Args:
        _resnet: `keras.Model`. ResNet model instance.
        block_num: int, block number to extract.

    Returns:
        A Keras Model representing the specified ResNet block.
    """

    extractor_levels = ["P2", "P3", "P4", "P5"]
    num_blocks = _resnet.stackwise_num_blocks
    if block_num == 0:
        x = _resnet.get_layer("pool1_pool").output
    else:
        x = _resnet.pyramid_outputs[extractor_levels[block_num - 1]]
    y = _resnet.get_layer(
        f"stack{block_num}_block{num_blocks[block_num]-1}_add"
    ).output
    return keras.models.Model(
        inputs=x,
        outputs=y,
        name=f"resnet_block{block_num + 1}",
    )


def basnet_predict(x_input, backbone, filters, segmentation_heads, dtype=None):
    """BASNet Prediction Module.

    This module outputs a coarse label map by integrating heavy
    encoder, bridge, and decoder blocks.

    Args:
        x_input: Input keras tensor.
        backbone: `keras.Model`. The backbone network used as a feature
            extractor for BASNet prediction encoder.
        filters: int, the number of filters.
        segmentation_heads: List of `keras.layers.Layer`, A list of Keras
            layers serving as the segmentation head for prediction module.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.


    Returns:
        A Keras Model that integrates the encoder, bridge, and decoder
        blocks for coarse label map prediction.
    """
    num_stages = 6

    x = x_input

    # -------------Encoder--------------
    x = keras.layers.Conv2D(
        filters, kernel_size=(3, 3), padding="same", dtype=dtype
    )(x)

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet backbone.
            x = get_resnet_block(backbone, i)(x)
            encoder_blocks.append(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = keras.layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), dtype=dtype
            )(x)
            for j in range(3):
                x = resnet_basic_block(
                    x,
                    filters=x.shape[3],
                    conv_shortcut=False,
                    name=f"v1_basic_block_{i + 1}_{j + 1}",
                    dtype=dtype,
                )
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2, dtype=dtype)
    x = convolution_block(x, filters=filters * 8, dilation=2, dtype=dtype)
    x = convolution_block(x, filters=filters * 8, dilation=2, dtype=dtype)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            x = keras.layers.UpSampling2D(
                size=2, interpolation="bilinear", dtype=dtype
            )(x)

        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8, dtype=dtype)
        x = convolution_block(x, filters=filters * 8, dtype=dtype)
        x = convolution_block(x, filters=filters * 8, dtype=dtype)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block)  # Prediction segmentation head.
        for segmentation_head, decoder_block in zip(
            segmentation_heads, decoder_blocks
        )
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


def basnet_rrm(base_model, filters, segmentation_head, dtype=None):
    """BASNet Residual Refinement Module (RRM).

    This module outputs a fine label map by integrating light encoder,
    bridge, and decoder blocks.

    Args:
        base_model: Keras model used as the base or coarse label map.
        filters: int, the number of filters.
        segmentation_head: a `keras.layers.Layer`, A Keras layer serving
            as the segmentation head for refinement module.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Returns:
        A Keras Model that constructs the Residual Refinement Module (RRM).
    """
    num_stages = 4

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = keras.layers.Conv2D(
        filters, kernel_size=(3, 3), padding="same", dtype=dtype
    )(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), dtype=dtype
        )(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters, dtype=dtype)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        x = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", dtype=dtype
        )(x)
        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x)  # Refinement segmentation head.

    # ------------- refined = coarse + residual
    x = keras.layers.Add(dtype=dtype)(
        [x_input, x]
    )  # Add prediction + refinement output

    return keras.models.Model(inputs=base_model.input, outputs=[x])
