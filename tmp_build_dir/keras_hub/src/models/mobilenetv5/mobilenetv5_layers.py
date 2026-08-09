import keras

from keras_hub.src.models.mobilenet.util import adjust_channels


class DropPath(keras.layers.Layer):
    """Implements the DropPath layer.

    DropPath is a form of stochastic depth, where connections are randomly
    dropped during training.

    Args:
        drop_prob: float. The probability of dropping a path.
        scale_by_keep: bool. If `True`, scale the output by `1/keep_prob`.
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=False):
        if self.drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + keras.random.uniform(
            shape, 0, 1, dtype=x.dtype
        )
        random_tensor = keras.ops.floor(random_tensor)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {"drop_prob": self.drop_prob, "scale_by_keep": self.scale_by_keep}
        )
        return config


class LayerScale2d(keras.layers.Layer):
    """A layer that applies a learnable scaling factor to the input tensor.

    This layer scales the input tensor by a learnable `gamma` parameter. The
    scaling is applied channel-wise.

    Args:
        dim: int. The number of channels in the input tensor.
        init_values: float. The initial value for the `gamma` parameter.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        dim,
        init_values=1e-5,
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.init_values = init_values
        self.data_format = data_format
        self.channel_axis = channel_axis

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )
        super().build(input_shape)

    def call(self, x):
        if self.data_format == "channels_first":
            gamma = keras.ops.reshape(self.gamma, (1, self.dim, 1, 1))
        else:
            gamma = keras.ops.reshape(self.gamma, (1, 1, 1, self.dim))
        return x * gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "init_values": self.init_values,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config


class RmsNorm2d(keras.layers.Layer):
    """A layer that applies Root Mean Square Normalization to a 2D input.

    This layer normalizes the input tensor along the channel dimension using
    the root mean square of the values, and then scales it by a learnable
    `gamma` parameter.

    Args:
        dim: int. The number of channels in the input tensor.
        eps: float. A small epsilon value to avoid division by zero.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        dim,
        eps=1e-6,
        data_format=None,
        channel_axis=None,
        gamma_initializer="ones",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.eps = eps
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.dim,),
            initializer=self.gamma_initializer,
            trainable=True,
            name="gamma",
        )
        super().build(input_shape)

    def call(self, x):
        input_dtype = x.dtype
        if self.data_format == "channels_first":
            x_permuted = keras.ops.transpose(x, (0, 2, 3, 1))
        else:
            x_permuted = x
        x_float = keras.ops.cast(x_permuted, "float32")
        norm_factor = keras.ops.rsqrt(
            keras.ops.mean(keras.ops.square(x_float), axis=-1, keepdims=True)
            + self.eps
        )
        norm_x_float = x_float * norm_factor
        norm_x = keras.ops.cast(norm_x_float, input_dtype)
        scaled_x = norm_x * self.gamma
        if self.data_format == "channels_first":
            output = keras.ops.transpose(scaled_x, (0, 3, 1, 2))
        else:
            output = scaled_x
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "eps": self.eps,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


class ConvNormAct(keras.layers.Layer):
    """A layer that combines convolution, normalization, and activation.

    This layer provides a convenient way to create a sequence of a 2D
    convolution, a normalization layer, and an activation function.

    Args:
        out_chs: int. The number of output channels.
        kernel_size: int or tuple. The size of the convolution kernel.
        stride: int or tuple. The stride of the convolution.
        dilation: int or tuple. The dilation rate of the convolution.
        groups: int. The number of groups for a grouped convolution.
        bias: bool. If `True`, a bias term is used in the convolution.
        pad_type: str. The type of padding to use. `"same"` or `""` for same
            padding, otherwise valid padding.
        apply_act: bool. If `True`, an activation function is applied.
        act_layer: str. The name of the activation function to use.
        norm_layer: str. The name of the normalization layer to use.
            Supported values are `"batch_norm"` and `"rms_norm"`.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        out_chs,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        pad_type="same",
        apply_act=True,
        act_layer="relu",
        norm_layer="batch_norm",
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.pad_type = pad_type
        self.apply_act = apply_act
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.bias_initializer = "zeros"
        padding_mode = "valid"
        if pad_type.lower() == "" or pad_type.lower() == "same":
            padding_mode = "same"

        self.conv = keras.layers.Conv2D(
            out_chs,
            kernel_size,
            strides=stride,
            padding=padding_mode,
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
        )

        if norm_layer == "batch_norm":
            self.norm = keras.layers.BatchNormalization(
                axis=self.channel_axis,
                epsilon=1e-5,
                gamma_initializer="ones",
                beta_initializer="zeros",
                dtype=self.dtype_policy,
            )
        elif norm_layer == "rms_norm":
            self.norm = RmsNorm2d(
                out_chs,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                gamma_initializer="ones",
                dtype=self.dtype_policy,
            )
        else:
            ln_axis = [1, 2, 3]
            if self.data_format == "channels_first":
                ln_axis = [2, 3, 1]
            self.norm = keras.layers.LayerNormalization(
                axis=ln_axis,
                dtype=self.dtype_policy,
            )

        if self.apply_act:
            if act_layer == "gelu":
                self.act = keras.layers.Activation(
                    lambda x: keras.activations.gelu(x, approximate=False),
                    dtype=self.dtype_policy,
                )
            else:
                self.act = keras.layers.Activation(
                    act_layer,
                    dtype=self.dtype_policy,
                )

    def build(self, input_shape):
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)
        if self.apply_act:
            self.act.build(conv_output_shape)
        self.built = True

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        if self.apply_act:
            x = self.act(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_chs": self.out_chs,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "dilation": self.dilation,
                "groups": self.groups,
                "bias": self.bias,
                "pad_type": self.pad_type,
                "apply_act": self.apply_act,
                "act_layer": self.act_layer,
                "norm_layer": self.norm_layer,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config


class SEModule(keras.layers.Layer):
    """Implements the Squeeze-and-Excitation (SE) module.

    The SE module adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.

    Args:
        channels: int. The number of input channels.
        rd_ratio: float. The reduction ratio for the bottleneck channels.
        rd_channels: int. The number of bottleneck channels. If specified,
            `rd_ratio` is ignored.
        rd_divisor: int. The divisor for rounding the number of bottleneck
            channels.
        add_maxpool: bool. If `True`, max pooling is used in addition to
            average pooling for the squeeze operation.
        bias: bool. If `True`, bias terms are used in the fully connected
            layers.
        act_layer: str. The activation function for the bottleneck layer.
        norm_layer: str. The normalization layer to use.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
        gate_layer: str. The gating activation function.
    """

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        add_maxpool=False,
        bias=True,
        act_layer="relu",
        norm_layer=None,
        data_format=None,
        channel_axis=None,
        gate_layer="sigmoid",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.channels = channels
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = adjust_channels(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.rd_ratio = rd_ratio
        self.rd_channels = rd_channels
        self.rd_divisor = rd_divisor
        self.bias = bias
        self.act_layer_arg = act_layer
        self.kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.bias_initializer = "zeros"
        self.norm_layer_arg = norm_layer
        self.gate_layer_arg = gate_layer
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.mean_axis = [2, 3] if data_format == "channels_first" else [1, 2]
        self.fc1 = keras.layers.Conv2D(
            rd_channels,
            kernel_size=1,
            use_bias=bias,
            name="fc1",
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
        )
        self.bn = (
            keras.layers.BatchNormalization(
                axis=channel_axis, dtype=self.dtype_policy
            )
            if norm_layer
            else (lambda x, training: x)
        )
        self.act = keras.layers.Activation(act_layer, dtype=self.dtype_policy)
        self.fc2 = keras.layers.Conv2D(
            channels,
            kernel_size=1,
            use_bias=bias,
            name="fc2",
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
        )
        self.gate = keras.layers.Activation(gate_layer, dtype=self.dtype_policy)

    def build(self, input_shape):
        self.fc1.build(input_shape)
        fc1_output_shape = self.fc1.compute_output_shape(input_shape)
        if hasattr(self.bn, "build"):
            self.bn.build(fc1_output_shape)
        self.act.build(fc1_output_shape)
        self.fc2.build(fc1_output_shape)
        self.built = True

    def call(self, x, training=False):
        x_se = keras.ops.mean(x, axis=self.mean_axis, keepdims=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * keras.ops.max(
                x, axis=self.mean_axis, keepdims=True
            )
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se, training=training)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "rd_ratio": self.rd_ratio,
                "rd_channels": self.rd_channels,
                "rd_divisor": self.rd_divisor,
                "add_maxpool": self.add_maxpool,
                "bias": self.bias,
                "act_layer": self.act_layer_arg,
                "norm_layer": self.norm_layer_arg,
                "gate_layer": self.gate_layer_arg,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config
