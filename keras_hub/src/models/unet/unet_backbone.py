import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
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

        # Build encoder
        bottleneck, encoder_outputs = self._build_encoder(
            inputs=inputs,
            backbone=backbone,
            depth=depth,
            filters=filters,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            dtype=dtype,
        )

        # Build decoder
        x = self._build_decoder(
            inputs=inputs,
            bottleneck=bottleneck,
            encoder_outputs=encoder_outputs,
            backbone=backbone,
            filters=filters,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            upsampling_strategy=upsampling_strategy,
            use_residual=use_residual,
            use_attention=use_attention,
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            dtype=dtype,
        )

        outputs = x

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

    def _build_encoder(
        self,
        inputs,
        backbone=None,
        depth=4,
        filters=64,
        use_batch_norm=False,
        use_residual=False,
        kernel_initializer="he_normal",
        data_format="channels_last",
        dtype=None,
    ):
        """Build the encoder (downsampling path) of the U-Net.

        Args:
            inputs: Input tensor.
            backbone: Optional pretrained backbone to use as encoder.
            depth: Depth of the encoder when building from scratch.
            filters: Number of filters in first layer when building from
                scratch.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            kernel_initializer: Initializer for kernel weights.
            data_format: Data format for the layers.
            dtype: Data type for the layers.

        Returns:
            Tuple of (bottleneck_tensor, list_of_skip_connection_tensors).
        """
        if backbone is not None:
            # Use pretrained backbone as encoder
            # Extract pyramid outputs from backbone
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

            bottleneck = encoder_outputs[-1]  # Last feature is bottleneck
            skip_connections = encoder_outputs[
                :-1
            ]  # All but last are skip connections
        else:
            # Build encoder from scratch
            x = inputs
            skip_connections = []

            # Encoder (downsampling path)
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
                if (
                    i < depth - 1
                ):  # Save as skip connection (don't save bottleneck)
                    skip_connections.append(x)
                if i < depth - 1:  # Pool all but the last level
                    x = keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        data_format=data_format,
                        dtype=dtype,
                        name=f"encoder_pool_{i}",
                    )(x)
            # x is now the bottleneck
            bottleneck = x

        return bottleneck, skip_connections

    def _build_decoder(
        self,
        inputs,
        bottleneck,
        encoder_outputs,
        backbone=None,
        filters=64,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.3,
        upsampling_strategy="transpose",
        use_residual=False,
        use_attention=False,
        kernel_initializer="he_normal",
        data_format="channels_last",
        dtype=None,
    ):
        """Build the decoder (upsampling path) of the U-Net.

        Args:
            inputs: Input tensor (needed for spatial dimension comparison).
            bottleneck: Bottleneck tensor from encoder.
            encoder_outputs: List of skip connection tensors from encoder.
            backbone: Optional pretrained backbone (affects filter calculation).
            filters: Number of filters in first layer (used when backbone is
                None).
            use_batch_norm: Whether to use batch normalization.
            use_dropout: Whether to use dropout.
            dropout_rate: Dropout rate if use_dropout is True.
            upsampling_strategy: 'transpose' or 'interpolation'.
            use_residual: Whether to use residual connections.
            use_attention: Whether to use attention gates.
            kernel_initializer: Initializer for kernel weights.
            data_format: Data format for the layers.
            dtype: Data type for the layers.

        Returns:
            Output tensor after decoder path.
        """
        x = bottleneck

        # Process from deepest to shallowest
        for i in range(len(encoder_outputs) - 1, -1, -1):
            # Determine number of filters
            if backbone is not None:
                # Get filters from encoder output shape
                skip_shape = encoder_outputs[i].shape
                num_filters = (
                    skip_shape[-1]
                    if data_format == "channels_last"
                    else skip_shape[1]
                )
            else:
                num_filters = filters * (2**i)

            # Upsample
            if upsampling_strategy == "transpose":
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
            else:  # interpolation
                x = keras.layers.UpSampling2D(
                    size=(2, 2),
                    data_format=data_format,
                    interpolation="bilinear",
                    dtype=dtype,
                    name=f"decoder_upsample_{i}",
                )(x)
                x = keras.layers.Conv2D(
                    num_filters,
                    kernel_size=(3, 3),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"decoder_upsample_conv_{i}",
                )(x)

            # Skip connection from encoder
            skip_connection = encoder_outputs[i]

            # Apply attention gate if enabled
            if use_attention:
                skip_connection = self._attention_gate(
                    skip_connection,
                    x,
                    num_filters,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"attention_gate_{i}",
                )

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
                use_residual=use_residual,
                kernel_initializer=kernel_initializer,
                data_format=data_format,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )

        # If using pretrained backbone, may need additional upsampling stages
        # to match the original input resolution
        if backbone is not None:
            # Calculate how many more upsampling stages we need
            # The decoder upsamples len(encoder_outputs) times
            # But the first encoder output may already be downsampled
            #
            # For ResNet50 with 128x128 input:
            #   conv1_relu: 64x64 (1st skip, downsampled by 2)
            #   conv2: 32x32 (2nd skip)
            #   conv3: 16x16 (3rd skip)
            #   conv4: 8x8 (4th skip)
            #   conv5: 4x4 (bottleneck)
            # Decoder upsamples 4 times (from 4x4): 8, 16, 32, 64
            # Final is 64x64, but input was 128x128, so need 1 more 2x upsample

            # Determine if first skip connection is at full resolution or
            # downsampled by comparing its spatial dimensions to the input
            input_spatial_shape = (
                inputs.shape[1:-1]
                if data_format == "channels_last"
                else inputs.shape[2:]
            )
            first_skip_spatial_shape = (
                encoder_outputs[0].shape[1:-1]
                if data_format == "channels_last"
                else encoder_outputs[0].shape[2:]
            )

            first_skip_downsampled = False
            if (
                input_spatial_shape[0] is not None
                and first_skip_spatial_shape[0] is not None
            ):
                # Static shapes available - compare directly
                if first_skip_spatial_shape[0] < input_spatial_shape[0]:
                    first_skip_downsampled = True
            elif (
                self._backbone_feature_names
                and len(self._backbone_feature_names) > 0
            ):
                # Dynamic shapes - fallback to layer name heuristic
                # This is imperfect but works for common architectures
                first_layer_name = self._backbone_feature_names[0].lower()
                # Common patterns: conv1, block_1, stem have stride > 1
                first_skip_downsampled = any(
                    pattern in first_layer_name
                    for pattern in ["conv1", "block_1", "stem", "pool"]
                )

            # If first skip is downsampled, we need one extra upsampling stage
            additional_stages = 1 if first_skip_downsampled else 0

            if data_format == "channels_last":
                final_filters = x.shape[-1]
            else:
                final_filters = x.shape[1]

            # Apply additional upsampling stages if needed
            for stage_idx in range(additional_stages):
                if upsampling_strategy == "transpose":
                    x = keras.layers.Conv2DTranspose(
                        final_filters,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=kernel_initializer,
                        data_format=data_format,
                        dtype=dtype,
                        name=f"final_upsample_transpose_{stage_idx}",
                    )(x)
                else:
                    x = keras.layers.UpSampling2D(
                        size=(2, 2),
                        data_format=data_format,
                        interpolation="bilinear",
                        dtype=dtype,
                        name=f"final_upsample_{stage_idx}",
                    )(x)
                    x = keras.layers.Conv2D(
                        final_filters,
                        kernel_size=(3, 3),
                        padding="same",
                        kernel_initializer=kernel_initializer,
                        data_format=data_format,
                        dtype=dtype,
                        name=f"final_upsample_conv_{stage_idx}",
                    )(x)

        return x

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
        """Double convolution block used in UNet.

        Args:
            x: Input tensor.
            filters: Number of filters for the convolution layers.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to add residual connection.
            kernel_initializer: Initializer for the kernel weights.
            data_format: Data format for the layers.
            dtype: Data type for the layers.
            name: Name prefix for the layers.

        Returns:
            Output tensor after the double convolution.
        """
        shortcut = x

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

        # Add residual connection if enabled
        if use_residual:
            # Match dimensions if needed
            shortcut_channels = (
                shortcut.shape[-1]
                if data_format == "channels_last"
                else shortcut.shape[1]
            )
            if shortcut_channels != filters:
                shortcut = keras.layers.Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_residual_proj",
                )(shortcut)
            x = keras.layers.Add(dtype=dtype, name=f"{name}_residual_add")(
                [x, shortcut]
            )

        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu2")(
            x
        )

        return x

    def _attention_gate(
        self,
        skip_connection,
        gating_signal,
        filters,
        data_format="channels_last",
        dtype=None,
        name="attention_gate",
    ):
        """Attention gate for skip connections.

        Args:
            skip_connection: Skip connection from encoder.
            gating_signal: Gating signal from decoder.
            filters: Number of intermediate filters.
            data_format: Data format for the layers.
            dtype: Data type for the layers.
            name: Name prefix for the layers.

        Returns:
            Attention-weighted skip connection.
        """
        # Upsample gating signal to match skip connection spatial dimensions
        # if needed
        skip_shape = skip_connection.shape
        gate_shape = gating_signal.shape

        if data_format == "channels_last":
            skip_h, skip_w = skip_shape[1], skip_shape[2]
            gate_h, gate_w = gate_shape[1], gate_shape[2]
        else:
            skip_h, skip_w = skip_shape[2], skip_shape[3]
            gate_h, gate_w = gate_shape[2], gate_shape[3]

        # If spatial dimensions don't match, upsample gating signal
        if skip_h != gate_h or skip_w != gate_w:
            scale_factor = skip_h // gate_h if skip_h > gate_h else 1
            if scale_factor > 1:
                gating_signal = keras.layers.UpSampling2D(
                    size=(scale_factor, scale_factor),
                    data_format=data_format,
                    interpolation="bilinear",
                    dtype=dtype,
                    name=f"{name}_gate_upsample",
                )(gating_signal)

        # Transform skip connection
        skip = keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_skip_conv",
        )(skip_connection)

        # Transform gating signal
        gate = keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_gate_conv",
        )(gating_signal)

        # Add and activate
        x = keras.layers.Add(dtype=dtype, name=f"{name}_add")([skip, gate])
        x = keras.layers.Activation("relu", dtype=dtype, name=f"{name}_relu")(x)

        # Generate attention coefficients
        x = keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_attention_conv",
        )(x)
        attention = keras.layers.Activation(
            "sigmoid", dtype=dtype, name=f"{name}_sigmoid"
        )(x)

        # Apply attention to skip connection
        output = keras.layers.Multiply(dtype=dtype, name=f"{name}_multiply")(
            [skip_connection, attention]
        )

        return output

    def _extract_backbone_features(
        self, backbone, inputs, data_format="channels_last"
    ):
        """Extract multi-scale features from a standard Keras backbone.

        Args:
            backbone: A Keras model to use as encoder.
            inputs: Input tensor.
            data_format: Data format for the layers.

        Returns:
            Tuple of (feature_maps, feature_names) where feature_maps is a list
            of feature tensors and feature_names is a list of layer names.
        """
        # Common layer patterns for feature extraction (in order from shallow
        # to deep)
        layer_patterns = [
            # ResNet50
            [
                "conv1_relu",
                "conv2_block3_out",
                "conv3_block4_out",
                "conv4_block6_out",
                "conv5_block3_out",
            ],
            # ResNet101/152
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

        # Try to find matching layers
        available_layers = {layer.name: layer for layer in backbone.layers}
        feature_layers = []

        for pattern in layer_patterns:
            matching = [name for name in pattern if name in available_layers]
            if len(matching) >= 3:  # Need at least 3 feature levels
                feature_layers = matching
                break

        if not feature_layers:
            # Fallback: analyze layer output shapes to find downsampling points
            feature_layers = []
            prev_spatial_size = None

            for layer in backbone.layers:
                if hasattr(layer, "output_shape"):
                    shape = layer.output_shape
                    if isinstance(shape, tuple) and len(shape) >= 3:
                        # Get spatial dimensions based on data format
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

                        # Track when spatial dimensions decrease (downsampling)
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

            # Add the final layer
            if backbone.layers:
                feature_layers.append(backbone.layers[-1].name)

            # Take at most 5 layers, well distributed
            if len(feature_layers) > 5:
                indices = [i * (len(feature_layers) - 1) // 4 for i in range(5)]
                feature_layers = [feature_layers[i] for i in indices]

        # Ensure we have at least 3 layers
        if len(feature_layers) < 3:
            # Last resort: take evenly spaced layers
            n_layers = len(backbone.layers)
            if n_layers >= 3:
                indices = [n_layers // 4, n_layers // 2, n_layers - 1]
                feature_layers = [backbone.layers[i].name for i in indices]

        # Create feature extractor
        if feature_layers:
            try:
                outputs = [
                    backbone.get_layer(name).output for name in feature_layers
                ]
                feature_extractor = keras.Model(backbone.input, outputs)
                return feature_extractor(inputs), feature_layers
            except Exception:
                # If extraction fails, use backbone output
                return [backbone(inputs)], []

        # Ultimate fallback
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
        if config.get("backbone") is not None:
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        return cls(**config)
