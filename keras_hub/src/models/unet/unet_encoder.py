"""UNet Encoder implementation."""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.UNetEncoder")
class UNetEncoder(Backbone):
    """UNet Encoder for image feature extraction.

    The encoder implements the downsampling (contracting) path of the U-Net
    architecture. It can either build an encoder from scratch or use a
    pretrained backbone as the encoder.

    When building from scratch, the encoder consists of repeated application of:
    - Two 3x3 convolutions (unpadded convolutions), each followed by ReLU
    - A 2x2 max pooling operation with stride 2 for downsampling

    Supports:
    - Vanilla encoder (original U-Net paper)
    - ResNet-style residual connections
    - Batch normalization
    - Pretrained backbones (ResNet, MobileNet, EfficientNet, etc.)

    Args:
        backbone: optional `keras.Model`. A pretrained backbone to use as the
            encoder. If provided, uses this backbone's intermediate features.
            If `None`, builds encoder from scratch. Defaults to `None`.
        depth: int. The depth of the encoder when building from scratch,
            representing the number of downsampling steps. Ignored if
            `backbone` is provided. Defaults to 4.
        filters: int. The number of filters in the first convolutional layer
            when building from scratch. The number of filters doubles at each
            downsampling step. Ignored if `backbone` is provided.
            Defaults to 64.
        image_shape: optional shape tuple. If `None`, defaults to
            `(None, None, 3)` for `"channels_last"` data format or
            `(3, None, None)` for `"channels_first"` data format.
        data_format: `None` or str. Either `"channels_last"` or
            `"channels_first"`. Defaults to `None`.
        use_batch_norm: bool. Whether to use batch normalization in the
            convolutional blocks. Defaults to False.
        use_residual: bool. Whether to add residual connections within
            convolutional blocks (ResUNet variant). Defaults to False.
        kernel_initializer: str or initializer. Defaults to "he_normal".
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`.

    Example:
    ```python
    import keras_hub
    import numpy as np

    # Build encoder from scratch
    encoder = keras_hub.models.UNetEncoder(
        depth=4,
        filters=64,
        use_batch_norm=True,
        use_residual=True,
    )

    # Use pretrained backbone
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )
    encoder = keras_hub.models.UNetEncoder(backbone=backbone)

    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    features = encoder(images)  # Returns dict with bottleneck and skip
    # connections
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
        use_residual=False,
        kernel_initializer="he_normal",
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)

        if image_shape is None:
            if data_format == "channels_last":
                image_shape = (None, None, 3)
            else:
                image_shape = (3, None, None)

        # Validate image_shape
        if data_format == "channels_last":
            if image_shape[-1] != 3:
                raise ValueError(
                    "image_shape must have 3 channels in the last dimension "
                    f"for channels_last format. Received: {image_shape}"
                )
        else:
            if image_shape[0] != 3:
                raise ValueError(
                    "image_shape must have 3 channels in the first dimension "
                    f"for channels_first format. Received: {image_shape}"
                )

        # Build functional model
        inputs = keras.layers.Input(shape=image_shape)

        self._backbone_feature_names = None

        if backbone is not None:
            # Use pretrained backbone as encoder
            if hasattr(backbone, "pyramid_outputs"):
                # KerasHub backbones with pyramid_outputs
                feature_extractor = keras.Model(
                    backbone.inputs, backbone.pyramid_outputs
                )
                features = feature_extractor(inputs)
                encoder_outputs = [features[k] for k in sorted(features.keys())]
            else:
                # Standard Keras backbones - extract intermediate layers
                encoder_outputs, feature_names = (
                    self._extract_backbone_features(
                        backbone, inputs, data_format
                    )
                )
                self._backbone_feature_names = feature_names

            bottleneck = encoder_outputs[-1]
            skip_connections = encoder_outputs[:-1]
        else:
            # Build encoder from scratch
            x = inputs
            skip_connections = []

            for i in range(depth):
                num_filters = filters * (2**i)
                x = self._conv_block(
                    x,
                    num_filters,
                    use_batch_norm=use_batch_norm,
                    use_residual=use_residual,
                    kernel_initializer=kernel_initializer,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"encoder_block_{i}",
                )
                if i < depth - 1:
                    skip_connections.append(x)
                if i < depth - 1:
                    x = keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        data_format=data_format,
                        dtype=dtype,
                        name=f"encoder_pool_{i}",
                    )(x)
            bottleneck = x

        # Return outputs as a dict for clarity
        outputs = {
            "bottleneck": bottleneck,
            "skip_connections": skip_connections,
        }

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # Config
        self.backbone = backbone
        self.depth = depth
        self.filters = filters
        self.image_shape = image_shape
        self.data_format = data_format
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.kernel_initializer = kernel_initializer

    def _conv_block(
        self,
        x,
        filters,
        use_batch_norm=False,
        use_residual=False,
        kernel_initializer="he_normal",
        data_format="channels_last",
        dtype=None,
        name="conv_block",
    ):
        """Double convolution block."""
        shortcut = x

        # First conv
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
            x = keras.layers.BatchNormalization(
                axis=-1 if data_format == "channels_last" else 1,
                dtype=dtype,
                name=f"{name}_bn1",
            )(x)
        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu1")(
            x
        )

        # Second conv
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
            x = keras.layers.BatchNormalization(
                axis=-1 if data_format == "channels_last" else 1,
                dtype=dtype,
                name=f"{name}_bn2",
            )(x)

        # Residual connection
        if use_residual:
            shortcut_filters = (
                shortcut.shape[-1]
                if data_format == "channels_last"
                else shortcut.shape[1]
            )
            if shortcut_filters != filters:
                shortcut = keras.layers.Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_residual_projection",
                )(shortcut)
            x = keras.layers.Add(dtype=dtype, name=f"{name}_add")([x, shortcut])

        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu2")(
            x
        )
        return x

    def _extract_backbone_features(
        self, backbone, inputs, data_format="channels_last"
    ):
        """Extract multi-scale features from a standard Keras backbone."""
        layer_patterns = [
            # ResNet50/101/152
            [
                "conv1_relu",
                "conv2_block3_out",
                "conv3_block4_out",
                "conv4_block6_out",
                "conv5_block3_out",
            ],
            [
                "conv1_relu",
                "conv2_block3_out",
                "conv3_block4_out",
                "conv4_block23_out",
                "conv5_block3_out",
            ],
            # VGG16/19
            [
                "block1_pool",
                "block2_pool",
                "block3_pool",
                "block4_pool",
                "block5_pool",
            ],
            # EfficientNet
            [
                "block2a_expand_activation",
                "block3a_expand_activation",
                "block4a_expand_activation",
                "block6a_expand_activation",
                "top_activation",
            ],
            # MobileNetV2
            [
                "block_1_expand_relu",
                "block_3_expand_relu",
                "block_6_expand_relu",
                "block_13_expand_relu",
                "out_relu",
            ],
        ]

        available_layers = {layer.name: layer for layer in backbone.layers}
        feature_layers = []

        for pattern in layer_patterns:
            matching = [name for name in pattern if name in available_layers]
            if len(matching) >= 3:
                feature_layers = matching
                break

        if not feature_layers:
            # Fallback: analyze spatial dimensions
            feature_layers = []
            prev_spatial_size = None

            for layer in backbone.layers:
                if hasattr(layer, "output_shape"):
                    shape = layer.output_shape
                    if isinstance(shape, tuple) and len(shape) >= 3:
                        if data_format == "channels_last":
                            spatial_size = (
                                (shape[1], shape[2])
                                if shape[1] is not None
                                else None
                            )
                        else:
                            spatial_size = (
                                (shape[2], shape[3])
                                if len(shape) > 2 and shape[2] is not None
                                else None
                            )

                        if (
                            spatial_size is not None
                            and prev_spatial_size is not None
                        ):
                            if (
                                spatial_size[0] is not None
                                and prev_spatial_size[0] is not None
                                and spatial_size[0] < prev_spatial_size[0]
                            ):
                                feature_layers.append(layer.name)

                        if spatial_size is not None:
                            prev_spatial_size = spatial_size

            if backbone.layers:
                feature_layers.append(backbone.layers[-1].name)

            if len(feature_layers) > 5:
                indices = [i * (len(feature_layers) - 1) // 4 for i in range(5)]
                feature_layers = [feature_layers[i] for i in indices]

        if len(feature_layers) < 3:
            n_layers = len(backbone.layers)
            if n_layers >= 3:
                indices = [n_layers // 4, n_layers // 2, n_layers - 1]
                feature_layers = [backbone.layers[i].name for i in indices]

        if feature_layers:
            try:
                outputs = [
                    backbone.get_layer(name).output for name in feature_layers
                ]
                feature_extractor = keras.Model(backbone.input, outputs)
                return feature_extractor(inputs), feature_layers
            except Exception:
                return [backbone(inputs)], []

        return [backbone(inputs)], []

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
                "use_residual": self.use_residual,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("backbone") is not None:
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        return cls(**config)
