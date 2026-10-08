import keras

from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import DropPath
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import LayerScale2d
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import RmsNorm2d


class MultiQueryAttention2d(keras.layers.Layer):
    """Implements 2D Multi-Query Attention.

    This layer performs attention on 2D spatial inputs. It uses a multi-query
    attention mechanism where multiple query heads attend to a single key and
    value.

    Args:
        filters: int. The output channel dimension.
        num_heads: int. The number of attention heads.
        key_dim: int. The dimension of the key. If `None`, it is calculated as
            `dim // num_heads`.
        value_dim: int. The dimension of the value. If `None`, it is calculated
            as `dim // num_heads`.
        query_strides: int or tuple. The stride for downsampling the query.
        kv_stride: int. The stride for downsampling the key and value.
        dw_kernel_size: int. The kernel size for the depthwise convolution used
            for downsampling.
        dilation: int. The dilation rate for the depthwise convolution.
        padding: str. The padding type for convolutions.
        attn_drop: float. The dropout rate for the attention weights.
        proj_drop: float. The dropout rate for the output projection.
        norm_layer: keras.layers.Layer. The normalization layer to use.
        use_bias: bool. If `True`, bias terms are used in convolutions.
        channel_axis: int. The axis representing the channels in the input
            tensor.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
    """

    def __init__(
        self,
        filters,
        num_heads=8,
        key_dim=None,
        value_dim=None,
        query_strides=1,
        kv_stride=1,
        dw_kernel_size=3,
        dilation=1,
        padding="same",
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=keras.layers.BatchNormalization,
        use_bias=False,
        channel_axis=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.key_dim_arg = key_dim
        self.value_dim_arg = value_dim
        self.query_strides_arg = query_strides
        self.kv_stride = kv_stride
        self.dw_kernel_size = dw_kernel_size
        self.dilation = dilation
        self.padding_arg = padding
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.channel_axis = channel_axis
        self.data_format = data_format
        self.query_strides = (
            query_strides
            if isinstance(query_strides, (list, tuple))
            else (query_strides, query_strides)
        )
        self.has_query_strides = any([s > 1 for s in self.query_strides])
        self.padding = padding
        self.conv_kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.bias_initializer = "zeros"
        self.attn_drop_layer = keras.layers.Dropout(
            attn_drop, dtype=self.dtype_policy
        )

    def build(self, input_shape):
        super().build(input_shape)
        dim = input_shape[self.channel_axis]
        self.key_dim = self.key_dim_arg or dim // self.num_heads
        self.value_dim = self.value_dim_arg or dim // self.num_heads
        self.scale = self.key_dim**-0.5
        query_layers = []
        if self.has_query_strides:
            pool_padding = "valid" if self.padding == "valid" else "same"
            query_layers.append(
                keras.layers.AveragePooling2D(
                    pool_size=self.query_strides,
                    strides=self.query_strides,
                    padding=pool_padding,
                    data_format=self.data_format,
                    name="query_down_pool",
                    dtype=self.dtype_policy,
                )
            )
            if self.norm_layer is RmsNorm2d:
                norm = self.norm_layer(
                    dim=dim,
                    channel_axis=self.channel_axis,
                    data_format=self.data_format,
                    name="query_norm",
                    dtype=self.dtype_policy,
                )
            else:
                norm = self.norm_layer(
                    axis=self.channel_axis,
                    name="query_norm",
                    gamma_initializer="ones",
                    beta_initializer="zeros",
                    dtype=self.dtype_policy,
                )
            query_layers.append(norm)
        query_layers.append(
            keras.layers.Conv2D(
                filters=self.num_heads * self.key_dim,
                kernel_size=1,
                use_bias=self.use_bias,
                data_format=self.data_format,
                name="query_proj",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
        )
        self.query_layers = query_layers
        key_layers = []
        if self.kv_stride > 1:
            key_layers.append(
                keras.layers.DepthwiseConv2D(
                    kernel_size=self.dw_kernel_size,
                    strides=self.kv_stride,
                    dilation_rate=self.dilation,
                    padding=self.padding,
                    data_format=self.data_format,
                    name="key_down_conv",
                    depthwise_initializer=self.conv_kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    use_bias=False,
                    dtype=self.dtype_policy,
                )
            )
            if self.norm_layer is RmsNorm2d:
                norm = self.norm_layer(
                    dim=dim,
                    channel_axis=self.channel_axis,
                    data_format=self.data_format,
                    name="key_norm",
                    dtype=self.dtype_policy,
                )
            else:
                norm = self.norm_layer(
                    axis=self.channel_axis,
                    gamma_initializer="ones",
                    beta_initializer="zeros",
                    name="key_norm",
                    dtype=self.dtype_policy,
                )
            key_layers.append(norm)
        key_layers.append(
            keras.layers.Conv2D(
                filters=self.key_dim,
                kernel_size=1,
                padding="valid",
                use_bias=self.use_bias,
                data_format=self.data_format,
                name="key_proj",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
        )
        self.key_layers = key_layers
        value_layers = []
        if self.kv_stride > 1:
            value_layers.append(
                keras.layers.DepthwiseConv2D(
                    kernel_size=self.dw_kernel_size,
                    strides=self.kv_stride,
                    dilation_rate=self.dilation,
                    padding=self.padding,
                    data_format=self.data_format,
                    name="value_down_conv",
                    depthwise_initializer=self.conv_kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    use_bias=False,
                    dtype=self.dtype_policy,
                )
            )
            if self.norm_layer is RmsNorm2d:
                norm = self.norm_layer(
                    dim=dim,
                    channel_axis=self.channel_axis,
                    data_format=self.data_format,
                    name="value_norm",
                    dtype=self.dtype_policy,
                )
            else:
                norm = self.norm_layer(
                    axis=self.channel_axis,
                    gamma_initializer="ones",
                    beta_initializer="zeros",
                    name="value_norm",
                    dtype=self.dtype_policy,
                )
            value_layers.append(norm)
        value_layers.append(
            keras.layers.Conv2D(
                filters=self.value_dim,
                kernel_size=1,
                padding="valid",
                use_bias=self.use_bias,
                data_format=self.data_format,
                name="value_proj",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
        )
        self.value_layers = value_layers
        output_layers = []
        if self.has_query_strides:
            output_layers.append(
                keras.layers.UpSampling2D(
                    size=self.query_strides,
                    interpolation="bilinear",
                    data_format=self.data_format,
                    name="output_upsample",
                    dtype=self.dtype_policy,
                )
            )
        output_layers.append(
            keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                use_bias=self.use_bias,
                data_format=self.data_format,
                name="output_proj",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
        )
        output_layers.append(
            keras.layers.Dropout(self.proj_drop_rate, dtype=self.dtype_policy)
        )
        self.output_proj_layers = output_layers

    def call(self, x, training=False):
        B = keras.ops.shape(x)[0]
        q = x
        for layer in self.query_layers:
            try:
                q = layer(q, training=training)
            except TypeError:
                q = layer(q)
        k = x
        for layer in self.key_layers:
            try:
                k = layer(k, training=training)
            except TypeError:
                k = layer(k)
        v = x
        for layer in self.value_layers:
            try:
                v = layer(v, training=training)
            except TypeError:
                v = layer(v)
        if self.data_format == "channels_last":
            q = keras.ops.transpose(q, (0, 3, 1, 2))
            k = keras.ops.transpose(k, (0, 3, 1, 2))
            v = keras.ops.transpose(v, (0, 3, 1, 2))
        s_q = keras.ops.shape(q)
        h_q, w_q = s_q[2], s_q[3]
        q = keras.ops.reshape(q, (B, self.num_heads, self.key_dim, -1))
        q = keras.ops.transpose(q, (0, 1, 3, 2))
        k = keras.ops.reshape(k, (B, self.key_dim, -1))
        k = keras.ops.transpose(k, (0, 2, 1))
        k = keras.ops.expand_dims(k, axis=1)
        v = keras.ops.reshape(v, (B, self.value_dim, -1))
        v = keras.ops.transpose(v, (0, 2, 1))
        v = keras.ops.expand_dims(v, axis=1)
        q = q * self.scale
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        attn = keras.ops.softmax(attn, axis=-1)
        attn = self.attn_drop_layer(attn, training=training)
        o = keras.ops.matmul(attn, v)
        o = keras.ops.transpose(o, (0, 2, 1, 3))
        feat_dim = self.num_heads * self.value_dim
        o = keras.ops.reshape(o, (B, h_q, w_q, feat_dim))
        if self.data_format == "channels_first":
            o = keras.ops.transpose(o, (0, 3, 1, 2))
        x_out = o
        for layer in self.output_proj_layers:
            try:
                x_out = layer(x_out, training=training)
            except TypeError:
                x_out = layer(x_out)
        return x_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim_arg,
                "value_dim": self.value_dim_arg,
                "query_strides": self.query_strides_arg,
                "kv_stride": self.kv_stride,
                "dw_kernel_size": self.dw_kernel_size,
                "dilation": self.dilation,
                "padding": self.padding_arg,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "norm_layer": keras.saving.serialize_keras_object(
                    self.norm_layer
                ),
                "use_bias": self.use_bias,
                "channel_axis": self.channel_axis,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["norm_layer"] = keras.saving.deserialize_keras_object(
            config["norm_layer"]
        )
        return cls(**config)


class Attention2d(keras.layers.Layer):
    """Implements 2D Multi-Head Attention.

    This layer performs multi-head self-attention on 2D spatial inputs.

    Args:
        filters: int. The output channel dimension.
        num_heads: int. The number of attention heads.
        bias: bool. If `True`, bias terms are used in the qkv and projection
            convolutions.
        attn_drop: float. The dropout rate for the attention weights.
        proj_drop: float. The dropout rate for the output projection.
        channel_axis: int. The axis representing the channels in the input
            tensor.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
    """

    def __init__(
        self,
        filters,
        num_heads=32,
        bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        channel_axis=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.bias = bias
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        self.channel_axis = channel_axis
        self.data_format = data_format
        self.conv_kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.bias_initializer = "zeros"
        self.attn_drop_layer = keras.layers.Dropout(
            attn_drop, dtype=self.dtype_policy
        )

    def build(self, input_shape):
        super().build(input_shape)
        dim = input_shape[self.channel_axis]
        self.head_dim = dim // self.num_heads
        self.qkv = keras.layers.Conv2D(
            dim * 3,
            kernel_size=1,
            use_bias=self.bias,
            data_format=self.data_format,
            name="qkv",
            dtype=self.dtype_policy,
            kernel_initializer=self.conv_kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.proj = keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            use_bias=self.bias,
            data_format=self.data_format,
            name="proj",
            dtype=self.dtype_policy,
            kernel_initializer=self.conv_kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.proj_drop_layer = keras.layers.Dropout(
            self.proj_drop_rate, dtype=self.dtype_policy
        )

    def call(self, x, attn_mask=None, training=False):
        if self.data_format == "channels_first":
            B, C, H, W = keras.ops.shape(x)
        else:
            B, H, W, C = keras.ops.shape(x)
        qkv = self.qkv(x)
        if self.data_format == "channels_last":
            qkv = keras.ops.transpose(qkv, (0, 3, 1, 2))
        q, k, v = keras.ops.unstack(
            keras.ops.reshape(
                qkv,
                (B, 3, self.num_heads, self.head_dim, H * W),
            ),
            axis=1,
        )
        q = keras.ops.transpose(q, (0, 1, 3, 2))
        k = keras.ops.transpose(k, (0, 1, 2, 3))
        v = keras.ops.transpose(v, (0, 1, 3, 2))
        attn = keras.ops.matmul(q, k) * (self.head_dim**-0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = keras.ops.softmax(attn, axis=-1)
        attn = self.attn_drop_layer(attn, training=training)
        x = keras.ops.matmul(attn, v)
        x = keras.ops.transpose(x, (0, 1, 3, 2))
        if self.data_format == "channels_first":
            x = keras.ops.reshape(x, (B, -1, H, W))
        else:
            x = keras.ops.reshape(x, (B, H, W, -1))
        x = self.proj(x)
        x = self.proj_drop_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_heads": self.num_heads,
                "bias": self.bias,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "channel_axis": self.channel_axis,
                "data_format": self.data_format,
            }
        )
        return config


class MobileAttention(keras.layers.Layer):
    """MobileNetV5 attention block.

    This block combines attention with depthwise convolutions for efficiency.
    It can use either standard Multi-Head Attention or Multi-Query Attention.

    Args:
        filters: int. The number of output channels.
        stride: int. The stride for the block.
        dw_kernel_size: int. The kernel size for the depthwise convolution in
            Multi-Query Attention.
        dilation: int. The dilation rate for convolutions.
        pad_type: str. The padding type for convolutions.
        num_heads: int. The number of attention heads.
        key_dim: int. The dimension of the key.
        value_dim: int. The dimension of the value.
        use_multi_query: bool. If `True`, use `MultiQueryAttention2d`,
            otherwise use `Attention2d`.
        query_strides: tuple. The strides for the query downsampling.
        kv_stride: int. The stride for key/value downsampling.
        cpe_dw_kernel_size: int. The kernel size for the conditional position
            encoding depthwise convolution.
        noskip: bool. If `True`, the skip connection is disabled.
        norm_layer: str. The normalization layer to use (`"batch_norm"` or
            `"rms_norm"`).
        drop_path_rate: float. The stochastic depth rate.
        attn_drop: float. The dropout rate for the attention weights.
        proj_drop: float. The dropout rate for the output projection.
        layer_scale_init_value: float. The initial value for layer scale. If
            `None`, layer scale is not used.
        use_bias: bool. If `True`, bias terms are used in convolutions.
        use_cpe: bool. If `True`, a conditional position encoding is added.
        channel_axis: int. The axis representing the channels in the input
            tensor.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
    """

    def __init__(
        self,
        filters,
        stride=1,
        dw_kernel_size=3,
        dilation=1,
        pad_type="same",
        num_heads=8,
        key_dim=64,
        value_dim=64,
        use_multi_query=False,
        query_strides=(1, 1),
        kv_stride=1,
        cpe_dw_kernel_size=3,
        noskip=False,
        norm_layer="batch_norm",
        drop_path_rate=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        layer_scale_init_value=1e-5,
        use_bias=False,
        use_cpe=False,
        channel_axis=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.stride = stride
        self.dw_kernel_size = dw_kernel_size
        self.dilation = dilation
        self.pad_type = pad_type
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_multi_query = use_multi_query
        self.query_strides = query_strides
        self.kv_stride = kv_stride
        self.cpe_dw_kernel_size = cpe_dw_kernel_size
        self.noskip = noskip
        self.norm_layer_name = norm_layer
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        self.layer_scale_init_value = layer_scale_init_value
        self.use_bias = use_bias
        self.use_cpe = use_cpe
        self.channel_axis = channel_axis
        self.data_format = data_format
        self.conv_kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.bias_initializer = "zeros"

    def build(self, input_shape):
        super().build(input_shape)
        in_chs = input_shape[self.channel_axis]
        self.has_skip = (
            self.stride == 1 and in_chs == self.filters
        ) and not self.noskip
        if self.use_cpe:
            self.conv_cpe_dw = keras.layers.DepthwiseConv2D(
                kernel_size=self.cpe_dw_kernel_size,
                strides=1,
                padding="same",
                dilation_rate=self.dilation,
                use_bias=True,
                data_format=self.data_format,
                name="conv_cpe_dw",
                depthwise_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
        else:
            self.conv_cpe_dw = None
        if self.norm_layer_name == "batch_norm":
            self.norm = keras.layers.BatchNormalization(
                axis=self.channel_axis,
                name="norm",
                gamma_initializer="ones",
                beta_initializer="zeros",
                dtype=self.dtype_policy,
            )
        elif self.norm_layer_name == "rms_norm":
            self.norm = RmsNorm2d(
                in_chs,
                data_format=self.data_format,
                gamma_initializer="ones",
                channel_axis=self.channel_axis,
                name="norm",
                dtype=self.dtype_policy,
            )
        else:
            raise ValueError(f"Unsupported norm_layer: {self.norm_layer_name}")
        num_heads = self.num_heads
        if num_heads is None:
            assert in_chs % self.key_dim == 0
            num_heads = in_chs // self.key_dim
        attn_norm_layer = (
            RmsNorm2d
            if self.norm_layer_name == "rms_norm"
            else keras.layers.BatchNormalization
        )
        if self.use_multi_query:
            self.attn = MultiQueryAttention2d(
                filters=self.filters,
                num_heads=num_heads,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                query_strides=self.query_strides,
                kv_stride=self.kv_stride,
                dw_kernel_size=self.dw_kernel_size,
                dilation=self.dilation,
                padding=self.pad_type,
                attn_drop=self.attn_drop_rate,
                proj_drop=self.proj_drop_rate,
                norm_layer=attn_norm_layer,
                use_bias=self.use_bias,
                channel_axis=self.channel_axis,
                data_format=self.data_format,
                name="attn",
                dtype=self.dtype_policy,
            )
        else:
            self.attn = Attention2d(
                filters=self.filters,
                num_heads=num_heads,
                attn_drop=self.attn_drop_rate,
                proj_drop=self.proj_drop_rate,
                bias=self.use_bias,
                channel_axis=self.channel_axis,
                data_format=self.data_format,
                name="attn",
                dtype=self.dtype_policy,
            )
        if self.layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(
                self.filters,
                self.layer_scale_init_value,
                name="layer_scale",
                channel_axis=self.channel_axis,
                data_format=self.data_format,
                dtype=self.dtype_policy,
            )
        else:
            self.layer_scale = lambda x: x
        self.drop_path = (
            DropPath(self.drop_path_rate, dtype=self.dtype_policy)
            if self.drop_path_rate > 0.0
            else lambda x, training: x
        )

    def call(self, x, training=False):
        if self.conv_cpe_dw is not None:
            x = x + self.conv_cpe_dw(x)
        shortcut = x
        x_normed = self.norm(x, training=training)
        x_attn = self.attn(x_normed, training=training)
        x_scaled = self.layer_scale(x_attn)
        if self.has_skip:
            return self.drop_path(x_scaled, training=training) + shortcut
        else:
            return x_scaled

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "dw_kernel_size": self.dw_kernel_size,
                "dilation": self.dilation,
                "pad_type": self.pad_type,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "use_multi_query": self.use_multi_query,
                "query_strides": self.query_strides,
                "kv_stride": self.kv_stride,
                "cpe_dw_kernel_size": self.cpe_dw_kernel_size,
                "noskip": self.noskip,
                "norm_layer": self.norm_layer_name,
                "drop_path_rate": self.drop_path_rate,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_bias": self.use_bias,
                "use_cpe": self.use_cpe,
                "channel_axis": self.channel_axis,
                "data_format": self.data_format,
            }
        )
        return config
