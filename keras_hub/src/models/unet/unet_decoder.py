"""UNet Decoder implementation."""

import keras

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.UNetDecoder")
class UNetDecoder(keras.layers.Layer):
    """UNet Decoder for upsampling and feature reconstruction.

    The decoder implements the upsampling (expansive) path of the U-Net
    architecture. It takes the bottleneck features and skip connections from
    the encoder and progressively upsamples while merging features.

    At each upsampling step:
    - Upsample the feature map (either via Conv2DTranspose or UpSampling2D)
    - Optionally apply attention gates to skip connections
    - Concatenate with corresponding skip connection from encoder
    - Apply two 3x3 convolutions

    Supports:
    - Vanilla decoder (original U-Net paper)
    - Modern upsampling (bilinear + conv to avoid checkerboard artifacts)
    - Attention gates (Attention U-Net)
    - ResNet-style residual connections
    - Batch normalization and dropout

    Args:
        filters: int or None. Base number of filters. If `None`, filters are
            inferred from skip connection shapes. Defaults to `None`.
        depth: int or None. Depth of the decoder. If `None`, inferred from
            number of skip connections. Defaults to `None`.
        use_batch_norm: bool. Whether to use batch normalization.
            Defaults to False.
        use_dropout: bool. Whether to use dropout. Defaults to False.
        dropout_rate: float. Dropout rate if use_dropout is True.
            Defaults to 0.3.
        upsampling_strategy: str. Either `"transpose"` (Conv2DTranspose) or
            `"interpolation"` (UpSampling2D + Conv2D).
            Defaults to `"transpose"`.
        use_residual: bool. Whether to use residual connections within
            convolutional blocks. Defaults to False.
        use_attention: bool. Whether to apply attention gates to skip
            connections. Defaults to False.
        kernel_initializer: str or initializer. Defaults to "he_normal".
        data_format: str. Either `"channels_last"` or `"channels_first"`.
            Defaults to `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`.

    Example:
    ```python
    import keras_hub
    import numpy as np
    import keras

    # Create encoder
    encoder = keras_hub.models.UNetEncoder(depth=4, filters=64)

    # Create decoder with attention gates
    decoder = keras_hub.layers.UNetDecoder(
        filters=64,
        use_attention=True,
        upsampling_strategy="interpolation",
    )

    # Process image
    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    encoder_features = encoder(images)
    output = decoder(encoder_features)
    ```
    """

    def __init__(
        self,
        filters=None,
        depth=None,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.3,
        upsampling_strategy="transpose",
        use_residual=False,
        use_attention=False,
        kernel_initializer="he_normal",
        data_format="channels_last",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        if upsampling_strategy not in ["transpose", "interpolation"]:
            raise ValueError(
                f"upsampling_strategy must be 'transpose' or 'interpolation'. "
                f"Received: {upsampling_strategy}"
            )

        self.filters = filters
        self.depth = depth
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.upsampling_strategy = upsampling_strategy
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.kernel_initializer = kernel_initializer
        self.data_format = data_format

        self._decoder_blocks = []
        self._upsample_layers = []
        self._attention_gates = []
        self._concat_layers = []
        self._dropout_layers = []

    def build(self, input_shape):
        # Input is a dict with 'bottleneck' and 'skip_connections'
        if not isinstance(input_shape, dict):
            raise ValueError(
                "UNetDecoder expects input to be a dict with keys "
                "'bottleneck' and 'skip_connections'"
            )

        skip_shapes = input_shape["skip_connections"]

        num_skip_connections = len(skip_shapes)

        # Build decoder layers for each upsampling stage
        for i in range(num_skip_connections - 1, -1, -1):
            skip_shape = skip_shapes[i]

            # Determine number of filters
            if self.filters is not None:
                num_filters = self.filters * (2**i)
            else:
                # Infer from skip connection shape
                num_filters = (
                    skip_shape[-1]
                    if self.data_format == "channels_last"
                    else skip_shape[1]
                )

            # Upsample layer
            if self.upsampling_strategy == "transpose":
                upsample = keras.layers.Conv2DTranspose(
                    num_filters,
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    data_format=self.data_format,
                    dtype=self.dtype,
                    name=f"decoder_upsample_{i}",
                )
            else:
                upsample = [
                    keras.layers.UpSampling2D(
                        size=(2, 2),
                        data_format=self.data_format,
                        interpolation="bilinear",
                        dtype=self.dtype,
                        name=f"decoder_upsample_{i}",
                    ),
                    keras.layers.Conv2D(
                        num_filters,
                        kernel_size=(3, 3),
                        padding="same",
                        kernel_initializer=self.kernel_initializer,
                        data_format=self.data_format,
                        dtype=self.dtype,
                        name=f"decoder_upsample_conv_{i}",
                    ),
                ]
            self._upsample_layers.append(upsample)

            # Attention gate
            if self.use_attention:
                attention_gate = self._build_attention_gate(
                    num_filters, name=f"attention_gate_{i}"
                )
                self._attention_gates.append(attention_gate)
            else:
                self._attention_gates.append(None)

            # Concatenate layer
            concat = keras.layers.Concatenate(
                axis=-1 if self.data_format == "channels_last" else 1,
                dtype=self.dtype,
                name=f"decoder_concat_{i}",
            )
            self._concat_layers.append(concat)

            # Dropout layer
            if self.use_dropout and i > 0:
                dropout = keras.layers.Dropout(
                    self.dropout_rate,
                    dtype=self.dtype,
                    name=f"decoder_dropout_{i}",
                )
                self._dropout_layers.append(dropout)
            else:
                self._dropout_layers.append(None)

            # Convolutional block
            conv_block = self._build_conv_block(
                num_filters, name=f"decoder_block_{i}"
            )
            self._decoder_blocks.append(conv_block)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass of the decoder.

        Args:
            inputs: Dict with keys:
                - 'bottleneck': Bottleneck tensor from encoder
                - 'skip_connections': List of skip connection tensors
            training: Boolean, whether in training mode.

        Returns:
            Output tensor after decoder path.
        """
        bottleneck = inputs["bottleneck"]
        skip_connections = inputs["skip_connections"]

        x = bottleneck

        # Process from deepest to shallowest
        for i, (
            upsample,
            attention_gate,
            concat,
            dropout,
            conv_block,
        ) in enumerate(
            zip(
                self._upsample_layers,
                self._attention_gates,
                self._concat_layers,
                self._dropout_layers,
                self._decoder_blocks,
            )
        ):
            # Upsample
            if isinstance(upsample, list):
                for layer in upsample:
                    x = layer(x)
            else:
                x = upsample(x)

            # Get skip connection (reversed order)
            skip_idx = len(skip_connections) - 1 - i
            skip_connection = skip_connections[skip_idx]

            # Apply attention gate if enabled
            if attention_gate is not None:
                skip_connection = attention_gate([skip_connection, x])

            # Concatenate
            x = concat([x, skip_connection])

            # Dropout
            if dropout is not None:
                x = dropout(x, training=training)

            # Convolutional block with optional residual connection
            x_pre_conv = x
            x = conv_block(x)

            # Add residual connection if enabled
            if self.use_residual:
                shortcut_channels = (
                    x_pre_conv.shape[-1]
                    if self.data_format == "channels_last"
                    else x_pre_conv.shape[1]
                )
                # Get target filters from conv block output
                target_filters = (
                    x.shape[-1]
                    if self.data_format == "channels_last"
                    else x.shape[1]
                )
                if shortcut_channels != target_filters:
                    x_pre_conv = keras.layers.Conv2D(
                        target_filters,
                        kernel_size=(1, 1),
                        padding="same",
                        kernel_initializer=self.kernel_initializer,
                        data_format=self.data_format,
                        dtype=self.dtype,
                        name=f"{conv_block.name}_residual_proj",
                    )(x_pre_conv)
                x = keras.layers.Add(
                    dtype=self.dtype, name=f"{conv_block.name}_residual_add"
                )([x, x_pre_conv])

        return x

    def _build_conv_block(self, filters, name="conv_block"):
        """Build a double convolution block."""
        layers_list = []

        # First conv
        layers_list.append(
            keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=self.kernel_initializer,
                data_format=self.data_format,
                dtype=self.dtype,
                name=f"{name}_conv1",
            )
        )
        if self.use_batch_norm:
            layers_list.append(
                keras.layers.BatchNormalization(
                    axis=-1 if self.data_format == "channels_last" else 1,
                    dtype=self.dtype,
                    name=f"{name}_bn1",
                )
            )
        layers_list.append(
            keras.layers.Activation(
                "relu", dtype=self.dtype, name=f"{name}_relu1"
            )
        )

        # Second conv
        layers_list.append(
            keras.layers.Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=self.kernel_initializer,
                data_format=self.data_format,
                dtype=self.dtype,
                name=f"{name}_conv2",
            )
        )
        if self.use_batch_norm:
            layers_list.append(
                keras.layers.BatchNormalization(
                    axis=-1 if self.data_format == "channels_last" else 1,
                    dtype=self.dtype,
                    name=f"{name}_bn2",
                )
            )

        layers_list.append(
            keras.layers.Activation(
                "relu", dtype=self.dtype, name=f"{name}_relu2"
            )
        )

        return keras.Sequential(layers_list, name=name)

    def _build_attention_gate(self, filters, name="attention_gate"):
        """Build an attention gate for skip connections."""
        # This is a simplified attention gate
        # In practice, this would be more complex
        layers_dict = {}

        # Gating signal processing
        layers_dict["g_conv"] = keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=self.kernel_initializer,
            data_format=self.data_format,
            dtype=self.dtype,
            name=f"{name}_g_conv",
        )

        # Skip connection processing
        layers_dict["x_conv"] = keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=self.kernel_initializer,
            data_format=self.data_format,
            dtype=self.dtype,
            name=f"{name}_x_conv",
        )

        # Attention computation
        layers_dict["psi"] = keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=self.kernel_initializer,
            data_format=self.data_format,
            dtype=self.dtype,
            name=f"{name}_psi",
        )

        # Build as a functional model
        def attention_gate_fn(inputs_list):
            skip_connection, gating_signal = inputs_list

            # Process skip connection
            x_processed = layers_dict["x_conv"](skip_connection)

            # Process gating signal (needs upsampling if shapes don't match)
            g_processed = layers_dict["g_conv"](gating_signal)

            # Match spatial dimensions if needed
            if self.data_format == "channels_last":
                skip_h, skip_w = (
                    skip_connection.shape[1],
                    skip_connection.shape[2],
                )
                gate_h, gate_w = g_processed.shape[1], g_processed.shape[2]
            else:
                skip_h, skip_w = (
                    skip_connection.shape[2],
                    skip_connection.shape[3],
                )
                gate_h, gate_w = g_processed.shape[2], g_processed.shape[3]

            # If shapes don't match, interpolate gating signal
            if skip_h != gate_h or skip_w != gate_w:
                if skip_h is not None and gate_h is not None:
                    scale_factor = skip_h // gate_h if gate_h else 1
                    if scale_factor > 1:
                        g_processed = keras.layers.UpSampling2D(
                            size=(scale_factor, scale_factor),
                            data_format=self.data_format,
                            interpolation="bilinear",
                            dtype=self.dtype,
                        )(g_processed)

            # Add and activate
            combined = keras.layers.Add(dtype=self.dtype)(
                [x_processed, g_processed]
            )
            combined = keras.layers.Activation("relu", dtype=self.dtype)(
                combined
            )

            # Compute attention coefficients
            attention = layers_dict["psi"](combined)
            attention = keras.layers.Activation("sigmoid", dtype=self.dtype)(
                attention
            )

            # Apply attention
            output = keras.layers.Multiply(dtype=self.dtype)(
                [skip_connection, attention]
            )
            return output

        return attention_gate_fn

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "depth": self.depth,
                "use_batch_norm": self.use_batch_norm,
                "use_dropout": self.use_dropout,
                "dropout_rate": self.dropout_rate,
                "upsampling_strategy": self.upsampling_strategy,
                "use_residual": self.use_residual,
                "use_attention": self.use_attention,
                "kernel_initializer": self.kernel_initializer,
                "data_format": self.data_format,
            }
        )
        return config
