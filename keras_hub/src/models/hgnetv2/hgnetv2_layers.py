import keras


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2LearnableAffineBlock(keras.layers.Layer):
    """
    HGNetV2 learnable affine block.

    Applies a learnable scale and bias to the input tensor, implementing a
    simple affine transformation with trainable parameters.

    Args:
        scale_value: float, optional. Initial value for the scale parameter.
            Defaults to 1.0.
        bias_value: float, optional. Initial value for the bias parameter.
            Defaults to 0.0.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, scale_value=1.0, bias_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_value = scale_value
        self.bias_value = bias_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self.scale_value),
            trainable=True,
            dtype=self.dtype,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(),
            initializer=keras.initializers.Constant(self.bias_value),
            trainable=True,
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, hidden_state):
        return self.scale * hidden_state + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {"scale_value": self.scale_value, "bias_value": self.bias_value}
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2ConvLayer(keras.layers.Layer):
    """
    HGNetV2 convolutional layer.

    Performs a 2D convolution followed by batch normalization and an activation
    function. Includes zero-padding to maintain spatial dimensions and
    optionally applies a learnable affine block.

    Args:
        in_channels: int. Number of input channels.
        out_channels: int. Number of output channels.
        kernel_size: int. Size of the convolutional kernel.
        stride: int. Stride of the convolution.
        groups: int. Number of groups for group convolution.
        activation: string, optional. Activation function to use ('relu',
            'gelu', 'tanh', or None). Defaults to 'relu'.
        use_learnable_affine_block: bool, optional. Whether to include a
            learnable affine block after activation. Defaults to False.
        data_format: string, optional. Data format of the input ('channels_last'
            or 'channels_first'). Defaults to None.
        channel_axis: int, optional. Axis of the channel dimension. Defaults to
            None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        activation="relu",
        use_learnable_affine_block=False,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.activation_name = activation
        self.use_learnable_affine_block = use_learnable_affine_block
        self.data_format = data_format
        self.channel_axis = channel_axis
        pad = (self.kernel_size - 1) // 2
        self.padding = keras.layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)),
            data_format=self.data_format,
            name=f"{self.name}_pad" if self.name else None,
        )
        self.convolution = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            groups=self.groups,
            padding="valid",
            use_bias=False,
            data_format=self.data_format,
            name=f"{self.name}_conv" if self.name else None,
            dtype=self.dtype_policy,
        )
        self.normalization = keras.layers.BatchNormalization(
            axis=self.channel_axis,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{self.name}_bn" if self.name else None,
            dtype=self.dtype_policy,
        )

        if self.activation_name == "relu":
            self.activation_layer = keras.layers.ReLU(
                name=f"{self.name}_relu" if self.name else None,
                dtype=self.dtype_policy,
            )
        elif self.activation_name == "gelu":
            self.activation_layer = keras.layers.Activation(
                "gelu",
                name=f"{self.name}_gelu" if self.name else None,
                dtype=self.dtype_policy,
            )
        elif self.activation_name == "tanh":
            self.activation_layer = keras.layers.Activation(
                "tanh",
                name=f"{self.name}_tanh" if self.name else None,
                dtype=self.dtype_policy,
            )
        elif self.activation_name is None:
            self.activation_layer = keras.layers.Identity(
                name=f"{self.name}_identity_activation" if self.name else None,
                dtype=self.dtype_policy,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        if self.use_learnable_affine_block:
            self.lab = HGNetV2LearnableAffineBlock(
                name=f"{self.name}_lab" if self.name else None,
                dtype=self.dtype_policy,
            )
        else:
            self.lab = keras.layers.Identity(
                name=f"{self.name}_identity_lab" if self.name else None
            )

    def build(self, input_shape):
        super().build(input_shape)
        self.padding.build(input_shape)
        padded_shape = self.padding.compute_output_shape(input_shape)
        self.convolution.build(padded_shape)
        conv_output_shape = self.convolution.compute_output_shape(padded_shape)
        self.normalization.build(conv_output_shape)
        self.lab.build(conv_output_shape)

    def call(self, inputs, training=None):
        hidden_state = self.padding(inputs)
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation_layer(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state

    def compute_output_shape(self, input_shape):
        padded_shape = self.padding.compute_output_shape(input_shape)
        shape = self.convolution.compute_output_shape(padded_shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "groups": self.groups,
                "activation": self.activation_name,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2ConvLayerLight(keras.layers.Layer):
    """
    HGNetV2 lightweight convolutional layer.

    Composes two convolutional layers: a 1x1 convolution followed by a depthwise
    convolution with the specified kernel size. Optionally includes a learnable
    affine block in the second convolution.

    Args:
        in_channels: int. Number of input channels.
        out_channels: int. Number of output channels.
        kernel_size: int. Size of the convolutional kernel for the depthwise
            convolution.
        use_learnable_affine_block: bool, optional. Whether to include a
            learnable affine block in the second convolution. Defaults to False.
        data_format: string, optional. Data format of the input. Defaults to
            None.
        channel_axis: int, optional. Axis of the channel dimension. Defaults to
            None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_learnable_affine_block=False,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_learnable_affine_block = use_learnable_affine_block
        self.data_format = data_format
        self.channel_axis = channel_axis

        self.conv1_layer = HGNetV2ConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            activation=None,
            use_learnable_affine_block=False,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_conv1" if self.name else "conv1",
            dtype=self.dtype_policy,
        )
        self.conv2_layer = HGNetV2ConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.out_channels,
            activation="relu",
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_conv2" if self.name else "conv2",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.conv1_layer.build(input_shape)
        conv1_output_shape = self.conv1_layer.compute_output_shape(input_shape)
        self.conv2_layer.build(conv1_output_shape)

    def call(self, hidden_state, training=None):
        hidden_state = self.conv1_layer(hidden_state, training=training)
        hidden_state = self.conv2_layer(hidden_state, training=training)
        return hidden_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        shape = self.conv1_layer.compute_output_shape(input_shape)
        shape = self.conv2_layer.compute_output_shape(shape)
        return shape


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2Embeddings(keras.layers.Layer):
    """
    HGNetV2 embedding layer.

    Processes input images through a series of convolutional and pooling
    operations to produce feature maps. Includes multiple convolutional layers
    with specific configurations, padding, and concatenation.

    Args:
        stem_channels: list of int. Channels for the stem layers.
        hidden_act: string. Activation function to use in the convolutional
            layers.
        use_learnable_affine_block: bool. Whether to use learnable affine blocks
            in the convolutional layers.
        data_format: string, optional. Data format of the input. Defaults to
            None.
        channel_axis: int, optional. Axis of the channel dimension. Defaults to
            None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        stem_channels,
        hidden_act,
        use_learnable_affine_block,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels
        self.hidden_act = hidden_act
        self.use_learnable_affine_block = use_learnable_affine_block
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.stem1_layer = HGNetV2ConvLayer(
            in_channels=self.stem_channels[0],
            out_channels=self.stem_channels[1],
            kernel_size=3,
            stride=2,
            groups=1,
            activation=self.hidden_act,
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_stem1" if self.name else "stem1",
            dtype=self.dtype_policy,
        )
        self.padding1 = keras.layers.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            data_format=self.data_format,
            name=f"{self.name}_padding1" if self.name else "padding1",
        )
        self.stem2a_layer = HGNetV2ConvLayer(
            in_channels=self.stem_channels[1],
            out_channels=self.stem_channels[1] // 2,
            kernel_size=2,
            stride=1,
            groups=1,
            activation=self.hidden_act,
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_stem2a" if self.name else "stem2a",
            dtype=self.dtype_policy,
        )
        self.padding2 = keras.layers.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            data_format=self.data_format,
            name=f"{self.name}_padding2" if self.name else "padding2",
        )
        self.stem2b_layer = HGNetV2ConvLayer(
            in_channels=self.stem_channels[1] // 2,
            out_channels=self.stem_channels[1],
            kernel_size=2,
            stride=1,
            groups=1,
            activation=self.hidden_act,
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_stem2b" if self.name else "stem2b",
            dtype=self.dtype_policy,
        )
        self.pool_layer = keras.layers.MaxPool2D(
            pool_size=2,
            strides=1,
            padding="valid",
            data_format=self.data_format,
            name=f"{self.name}_pool" if self.name else "pool",
        )
        self.concatenate_layer = keras.layers.Concatenate(
            axis=self.channel_axis,
            name=f"{self.name}_concat" if self.name else "concat",
        )
        self.stem3_layer = HGNetV2ConvLayer(
            in_channels=self.stem_channels[1] * 2,
            out_channels=self.stem_channels[1],
            kernel_size=3,
            stride=2,
            groups=1,
            activation=self.hidden_act,
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_stem3" if self.name else "stem3",
            dtype=self.dtype_policy,
        )
        self.stem4_layer = HGNetV2ConvLayer(
            in_channels=self.stem_channels[1],
            out_channels=self.stem_channels[2],
            kernel_size=1,
            stride=1,
            groups=1,
            activation=self.hidden_act,
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_stem4" if self.name else "stem4",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        super().build(input_shape)
        current_shape = input_shape
        self.stem1_layer.build(current_shape)
        current_shape = self.stem1_layer.compute_output_shape(current_shape)
        padded_shape1 = self.padding1.compute_output_shape(current_shape)
        self.stem2a_layer.build(padded_shape1)
        shape_after_stem2a = self.stem2a_layer.compute_output_shape(
            padded_shape1
        )
        padded_shape2 = self.padding2.compute_output_shape(shape_after_stem2a)
        self.stem2b_layer.build(padded_shape2)
        shape_after_stem2b = self.stem2b_layer.compute_output_shape(
            padded_shape2
        )
        shape_after_pool = self.pool_layer.compute_output_shape(padded_shape1)
        concat_input_shapes = [shape_after_pool, shape_after_stem2b]
        shape_after_concat = self.concatenate_layer.compute_output_shape(
            concat_input_shapes
        )
        self.stem3_layer.build(shape_after_concat)
        shape_after_stem3 = self.stem3_layer.compute_output_shape(
            shape_after_concat
        )
        self.stem4_layer.build(shape_after_stem3)

    def compute_output_shape(self, input_shape):
        current_shape = self.stem1_layer.compute_output_shape(input_shape)
        padded_shape1 = self.padding1.compute_output_shape(current_shape)
        shape_after_stem2a = self.stem2a_layer.compute_output_shape(
            padded_shape1
        )
        padded_shape2 = self.padding2.compute_output_shape(shape_after_stem2a)
        shape_after_stem2b = self.stem2b_layer.compute_output_shape(
            padded_shape2
        )
        shape_after_pool = self.pool_layer.compute_output_shape(padded_shape1)
        concat_input_shapes = [shape_after_pool, shape_after_stem2b]
        shape_after_concat = self.concatenate_layer.compute_output_shape(
            concat_input_shapes
        )
        shape_after_stem3 = self.stem3_layer.compute_output_shape(
            shape_after_concat
        )
        final_shape = self.stem4_layer.compute_output_shape(shape_after_stem3)
        return final_shape

    def call(self, pixel_values, training=None):
        embedding = self.stem1_layer(pixel_values, training=training)
        embedding_padded_for_2a_and_pool = self.padding1(embedding)
        emb_stem_2a = self.stem2a_layer(
            embedding_padded_for_2a_and_pool, training=training
        )
        emb_stem_2a_padded = self.padding2(emb_stem_2a)
        emb_stem_2a_processed = self.stem2b_layer(
            emb_stem_2a_padded, training=training
        )
        pooled_emb = self.pool_layer(embedding_padded_for_2a_and_pool)
        embedding_concatenated = self.concatenate_layer(
            [pooled_emb, emb_stem_2a_processed]
        )
        embedding_after_stem3 = self.stem3_layer(
            embedding_concatenated, training=training
        )
        final_embedding = self.stem4_layer(
            embedding_after_stem3, training=training
        )
        return final_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_channels": self.stem_channels,
                "hidden_act": self.hidden_act,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2BasicLayer(keras.layers.Layer):
    """
    HGNetV2 basic layer.

    Consists of multiple convolutional blocks followed by aggregation through
    concatenation and convolutional layers. Supports residual connections and
    drop path for regularization.

    Args:
        in_channels: int. Number of input channels.
        middle_channels: int. Number of channels in the intermediate
            convolutional blocks.
        out_channels: int. Number of output channels.
        layer_num: int. Number of convolutional blocks in the layer.
        kernel_size: int, optional. Kernel size for the convolutional blocks.
            Defaults to 3.
        residual: bool, optional. Whether to include a residual connection.
            Defaults to False.
        light_block: bool, optional. Whether to use lightweight convolutional
            blocks. Defaults to False.
        drop_path: float, optional. Drop path rate for regularization. Defaults
            to 0.0.
        use_learnable_affine_block: bool, optional. Whether to use learnable
            affine blocks in the convolutional blocks. Defaults to False.
        data_format: string, optional. Data format of the input. Defaults to
            None.
        channel_axis: int, optional. Axis of the channel dimension. Defaults to
            None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        in_channels,
        middle_channels,
        out_channels,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        drop_path=0.0,
        use_learnable_affine_block=False,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels_arg = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.layer_num = layer_num
        self.kernel_size = kernel_size
        self.residual = residual
        self.light_block = light_block
        self.drop_path_rate = drop_path
        self.use_learnable_affine_block = use_learnable_affine_block
        self.data_format = data_format
        self.channel_axis = channel_axis

        self.layer_list = []
        for i in range(self.layer_num):
            block_input_channels = (
                self.in_channels_arg if i == 0 else self.middle_channels
            )
            if self.light_block:
                block = HGNetV2ConvLayerLight(
                    in_channels=block_input_channels,
                    out_channels=self.middle_channels,
                    kernel_size=self.kernel_size,
                    use_learnable_affine_block=self.use_learnable_affine_block,
                    data_format=self.data_format,
                    channel_axis=self.channel_axis,
                    name=f"{self.name}_light_block_{i}"
                    if self.name
                    else f"light_block_{i}",
                    dtype=self.dtype_policy,
                )
            else:
                block = HGNetV2ConvLayer(
                    in_channels=block_input_channels,
                    out_channels=self.middle_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    groups=1,
                    activation="relu",
                    use_learnable_affine_block=self.use_learnable_affine_block,
                    data_format=self.data_format,
                    channel_axis=self.channel_axis,
                    name=f"{self.name}_conv_block_{i}"
                    if self.name
                    else f"conv_block_{i}",
                    dtype=self.dtype_policy,
                )
            self.layer_list.append(block)
        self.total_channels_for_aggregation = (
            self.in_channels_arg + self.layer_num * self.middle_channels
        )
        self.aggregation_squeeze_conv = HGNetV2ConvLayer(
            in_channels=self.total_channels_for_aggregation,
            out_channels=self.out_channels // 2,
            kernel_size=1,
            stride=1,
            groups=1,
            activation="relu",
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_agg_squeeze" if self.name else "agg_squeeze",
            dtype=self.dtype_policy,
        )
        self.aggregation_excitation_conv = HGNetV2ConvLayer(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            activation="relu",
            use_learnable_affine_block=self.use_learnable_affine_block,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            name=f"{self.name}_agg_excite" if self.name else "agg_excite",
            dtype=self.dtype_policy,
        )

        if self.drop_path_rate > 0.0:
            self.drop_path_layer = keras.layers.Dropout(
                self.drop_path_rate,
                noise_shape=(None, 1, 1, 1),
                name=f"{self.name}_drop_path" if self.name else "drop_path",
            )
        else:
            self.drop_path_layer = keras.layers.Identity(
                name=f"{self.name}_identity_drop_path"
                if self.name
                else "identity_drop_path"
            )

        self.concatenate_layer = keras.layers.Concatenate(
            axis=self.channel_axis,
            name=f"{self.name}_concat" if self.name else "concat",
        )
        if self.residual:
            self.add_layer = keras.layers.Add(
                name=f"{self.name}_add_residual"
                if self.name
                else "add_residual"
            )

    def build(self, input_shape):
        super().build(input_shape)
        current_block_input_shape = input_shape
        output_shapes_for_concat = [input_shape]
        for i, layer_block in enumerate(self.layer_list):
            layer_block.build(current_block_input_shape)
            current_block_output_shape = layer_block.compute_output_shape(
                current_block_input_shape
            )
            output_shapes_for_concat.append(current_block_output_shape)
            current_block_input_shape = current_block_output_shape
        concatenated_shape = self.concatenate_layer.compute_output_shape(
            output_shapes_for_concat
        )
        self.aggregation_squeeze_conv.build(concatenated_shape)
        agg_squeeze_output_shape = (
            self.aggregation_squeeze_conv.compute_output_shape(
                concatenated_shape
            )
        )
        self.aggregation_excitation_conv.build(agg_squeeze_output_shape)

    def compute_output_shape(self, input_shape):
        output_tensors_shapes = [input_shape]
        current_block_input_shape = input_shape
        for layer_block in self.layer_list:
            current_block_output_shape = layer_block.compute_output_shape(
                current_block_input_shape
            )
            output_tensors_shapes.append(current_block_output_shape)
            current_block_input_shape = current_block_output_shape
        concatenated_features_shape = (
            self.concatenate_layer.compute_output_shape(output_tensors_shapes)
        )
        aggregated_features_shape = (
            self.aggregation_squeeze_conv.compute_output_shape(
                concatenated_features_shape
            )
        )
        final_output_shape = (
            self.aggregation_excitation_conv.compute_output_shape(
                aggregated_features_shape
            )
        )

        return final_output_shape

    def call(self, hidden_state, training=None):
        identity = hidden_state
        output_tensors = [hidden_state]

        current_feature_map = hidden_state
        for layer_block in self.layer_list:
            current_feature_map = layer_block(
                current_feature_map, training=training
            )
            output_tensors.append(current_feature_map)
        concatenated_features = self.concatenate_layer(output_tensors)
        aggregated_features = self.aggregation_squeeze_conv(
            concatenated_features, training=training
        )
        aggregated_features = self.aggregation_excitation_conv(
            aggregated_features, training=training
        )
        if self.residual:
            dropped_features = self.drop_path_layer(
                aggregated_features, training=training
            )
            final_output = self.add_layer([dropped_features, identity])
        else:
            final_output = aggregated_features
        return final_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels_arg,
                "middle_channels": self.middle_channels,
                "out_channels": self.out_channels,
                "layer_num": self.layer_num,
                "kernel_size": self.kernel_size,
                "residual": self.residual,
                "light_block": self.light_block,
                "drop_path": self.drop_path_rate,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2Stage(keras.layers.Layer):
    """
    HGNetV2 stage layer.

    Represents a stage in the HGNetV2 model, which may include downsampling
    followed by a series of basic layers. Each stage can have different
    configurations for the number of blocks, channels, etc.

    Args:
        stage_in_channels: list of int. Input channels for each stage.
        stage_mid_channels: list of int. Middle channels for each stage.
        stage_out_channels: list of int. Output channels for each stage.
        stage_num_blocks: list of int. Number of basic layers in each stage.
        stage_num_of_layers: list of int. Number of convolutional blocks in
            each basic layer.
        apply_downsample: list of bools. Whether to downsample at the beginning
            of each stage.
        use_lightweight_conv_block: list of bools. Whether to use HGNetV2
            lightweight convolutional block in the stage.
        stage_kernel_size: list of int. Kernel sizes for each stage.
        use_learnable_affine_block: bool. Whether to use learnable affine
            blocks.
        stage_index: int. The index of the current stage.
        drop_path: float, optional. Drop path rate. Defaults to 0.0.
        data_format: string, optional. Data format of the input. Defaults to
            None.
        channel_axis: int, optional. Axis of the channel dimension. Defaults to
            None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        stage_in_channels,
        stage_mid_channels,
        stage_out_channels,
        stage_num_blocks,
        stage_num_of_layers,
        apply_downsample,
        use_lightweight_conv_block,
        stage_kernel_size,
        use_learnable_affine_block,
        stage_index: int,
        drop_path: float = 0.0,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_num_of_layers = stage_num_of_layers
        self.apply_downsample = apply_downsample
        self.use_lightweight_conv_block = use_lightweight_conv_block
        self.stage_kernel_size = stage_kernel_size
        self.use_learnable_affine_block = use_learnable_affine_block
        self.stage_index = stage_index
        self.drop_path = drop_path
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.current_stage_in_channels = stage_in_channels[stage_index]
        self.current_stage_mid_channels = stage_mid_channels[stage_index]
        self.current_stage_out_channels = stage_out_channels[stage_index]
        self.current_stage_num_blocks = stage_num_blocks[stage_index]
        self.current_stage_num_layers_per_block = stage_num_of_layers[
            stage_index
        ]
        self.current_stage_is_downsample_active = apply_downsample[stage_index]
        self.current_stage_is_light_block = use_lightweight_conv_block[
            stage_index
        ]
        self.current_stage_kernel_size = stage_kernel_size[stage_index]
        self.current_stage_use_lab = use_learnable_affine_block
        self.current_stage_drop_path = drop_path
        if self.current_stage_is_downsample_active:
            self.downsample_layer = HGNetV2ConvLayer(
                in_channels=self.current_stage_in_channels,
                out_channels=self.current_stage_in_channels,
                kernel_size=3,
                stride=2,
                groups=self.current_stage_in_channels,
                activation=None,
                use_learnable_affine_block=False,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                name=f"{self.name}_downsample" if self.name else "downsample",
                dtype=self.dtype_policy,
            )
        else:
            self.downsample_layer = keras.layers.Identity(
                name=f"{self.name}_identity_downsample"
                if self.name
                else "identity_downsample"
            )

        self.blocks_list = []
        for i in range(self.current_stage_num_blocks):
            basic_layer_input_channels = (
                self.current_stage_in_channels
                if i == 0
                else self.current_stage_out_channels
            )

            block = HGNetV2BasicLayer(
                in_channels=basic_layer_input_channels,
                middle_channels=self.current_stage_mid_channels,
                out_channels=self.current_stage_out_channels,
                layer_num=self.current_stage_num_layers_per_block,
                residual=(False if i == 0 else True),
                kernel_size=self.current_stage_kernel_size,
                light_block=self.current_stage_is_light_block,
                drop_path=self.current_stage_drop_path,
                use_learnable_affine_block=self.current_stage_use_lab,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                name=f"{self.name}_block_{i}" if self.name else f"block_{i}",
                dtype=self.dtype_policy,
            )
            self.blocks_list.append(block)

    def build(self, input_shape):
        super().build(input_shape)
        current_input_shape = input_shape
        self.downsample_layer.build(current_input_shape)
        current_input_shape = self.downsample_layer.compute_output_shape(
            current_input_shape
        )

        for block_item in self.blocks_list:
            block_item.build(current_input_shape)
            current_input_shape = block_item.compute_output_shape(
                current_input_shape
            )

    def compute_output_shape(self, input_shape):
        current_shape = self.downsample_layer.compute_output_shape(input_shape)
        for block_item in self.blocks_list:
            current_shape = block_item.compute_output_shape(current_shape)
        return current_shape

    def call(self, hidden_state, training=None):
        hidden_state = self.downsample_layer(hidden_state, training=training)
        for block_item in self.blocks_list:
            hidden_state = block_item(hidden_state, training=training)
        return hidden_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stage_in_channels": self.stage_in_channels,
                "stage_mid_channels": self.stage_mid_channels,
                "stage_out_channels": self.stage_out_channels,
                "stage_num_blocks": self.stage_num_blocks,
                "stage_num_of_layers": self.stage_num_of_layers,
                "apply_downsample": self.apply_downsample,
                "use_lightweight_conv_block": self.use_lightweight_conv_block,
                "stage_kernel_size": self.stage_kernel_size,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "stage_index": self.stage_index,
                "drop_path": self.drop_path,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config
