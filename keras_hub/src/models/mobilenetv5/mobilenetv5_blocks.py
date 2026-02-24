import keras

from keras_hub.src.models.mobilenet.util import adjust_channels
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import DropPath
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import LayerScale2d
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import RmsNorm2d
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import num_groups


class UniversalInvertedResidual(keras.layers.Layer):
    """Universal Inverted Residual block.

    This block is a flexible and universal version of the inverted residual
    block, which can be configured to behave like different variants of mobile
    convolutional blocks.

    Args:
        filters: int. The number of output channels.
        dw_kernel_size_start: int. The kernel size for the initial depthwise
            convolution. If 0, this layer is skipped.
        dw_kernel_size_mid: int. The kernel size for the middle depthwise
            convolution. If 0, this layer is skipped.
        dw_kernel_size_end: int. The kernel size for the final depthwise
            convolution. If 0, this layer is skipped.
        stride: int. The stride for the block.
        dilation: int. The dilation rate for convolutions.
        pad_type: str. The padding type for convolutions.
        noskip: bool. If `True`, the skip connection is disabled.
        exp_ratio: float. The expansion ratio for the middle channels.
        act_layer: str. The activation function to use.
        norm_layer: str. The normalization layer to use.
        se_layer: keras.layers.Layer. The Squeeze-and-Excitation layer to use.
        drop_path_rate: float. The stochastic depth rate.
        layer_scale_init_value: float. The initial value for layer scale. If
            `None`, layer scale is not used.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        filters,
        dw_kernel_size_start=0,
        dw_kernel_size_mid=3,
        dw_kernel_size_end=0,
        stride=1,
        dilation=1,
        pad_type="same",
        noskip=False,
        exp_ratio=1.0,
        act_layer="relu",
        norm_layer="batch_norm",
        se_layer=None,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.dw_kernel_size_start = dw_kernel_size_start
        self.dw_kernel_size_mid = dw_kernel_size_mid
        self.dw_kernel_size_end = dw_kernel_size_end
        self.stride = stride
        self.dilation = dilation
        self.pad_type = pad_type
        self.noskip = noskip
        self.exp_ratio = exp_ratio
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = se_layer
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.data_format = data_format
        self.channel_axis = channel_axis

    def build(self, input_shape):
        super().build(input_shape)
        in_chs = input_shape[self.channel_axis]
        self.has_skip = (
            in_chs == self.filters and self.stride == 1
        ) and not self.noskip
        use_bias = self.norm_layer == "rms_norm"

        if self.dw_kernel_size_start:
            self.dw_start = ConvNormAct(
                in_chs,
                self.dw_kernel_size_start,
                stride=self.stride if not self.dw_kernel_size_mid else 1,
                dilation=self.dilation,
                groups=in_chs,
                pad_type=self.pad_type,
                apply_act=False,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                bias=use_bias,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
        else:
            self.dw_start = lambda x, training=False: x

        mid_chs = adjust_channels(in_chs * self.exp_ratio)
        self.pw_exp = ConvNormAct(
            mid_chs,
            1,
            pad_type=self.pad_type,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
            bias=use_bias,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            dtype=self.dtype_policy,
        )

        if self.dw_kernel_size_mid:
            self.dw_mid = ConvNormAct(
                mid_chs,
                self.dw_kernel_size_mid,
                stride=self.stride,
                dilation=self.dilation,
                groups=mid_chs,
                pad_type=self.pad_type,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                bias=use_bias,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
        else:
            self.dw_mid = lambda x, training=False: x
        self.se = (
            self.se_layer(
                filters=mid_chs,
                bottleneck_filters=adjust_channels(mid_chs * 0.25),
                squeeze_activation=self.act_layer,
                excite_activation="sigmoid",
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
            if self.se_layer
            else (lambda x, training=False: x)
        )
        self.pw_proj = ConvNormAct(
            self.filters,
            1,
            pad_type=self.pad_type,
            apply_act=False,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
            bias=use_bias,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            dtype=self.dtype_policy,
        )

        if self.dw_kernel_size_end:
            self.dw_end = ConvNormAct(
                self.filters,
                self.dw_kernel_size_end,
                stride=self.stride
                if not self.dw_kernel_size_start and not self.dw_kernel_size_mid
                else 1,
                dilation=self.dilation,
                groups=self.filters,
                pad_type=self.pad_type,
                apply_act=False,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                bias=use_bias,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
        else:
            self.dw_end = lambda x, training=False: x

        self.layer_scale = (
            LayerScale2d(
                self.filters,
                self.layer_scale_init_value,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
            if self.layer_scale_init_value is not None
            else lambda x: x
        )
        self.drop_path = (
            DropPath(self.drop_path_rate, dtype=self.dtype_policy)
            if self.drop_path_rate > 0.0
            else (lambda x, training=False: x)
        )
        current_shape = input_shape
        if hasattr(self.dw_start, "build"):
            self.dw_start.build(current_shape)
            current_shape = self.dw_start.compute_output_shape(current_shape)
        self.pw_exp.build(current_shape)
        current_shape = self.pw_exp.compute_output_shape(current_shape)
        if hasattr(self.dw_mid, "build"):
            self.dw_mid.build(current_shape)
            current_shape = self.dw_mid.compute_output_shape(current_shape)
        if hasattr(self.se, "build"):
            self.se.build(current_shape)
        self.pw_proj.build(current_shape)
        current_shape = self.pw_proj.compute_output_shape(current_shape)
        if hasattr(self.dw_end, "build"):
            self.dw_end.build(current_shape)
            current_shape = self.dw_end.compute_output_shape(current_shape)
        if hasattr(self.layer_scale, "build"):
            self.layer_scale.build(current_shape)

    def call(self, x, training=False):
        shortcut = x
        x = self.dw_start(x, training=training)
        x = self.pw_exp(x, training=training)
        x = self.dw_mid(x, training=training)
        x = self.se(x, training=training)
        x = self.pw_proj(x, training=training)
        x = self.dw_end(x, training=training)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x, training=training) + shortcut
        return x

    def compute_output_shape(self, input_shape):
        current_shape = input_shape
        if hasattr(self.dw_start, "compute_output_shape"):
            current_shape = self.dw_start.compute_output_shape(current_shape)
        current_shape = self.pw_exp.compute_output_shape(current_shape)
        if hasattr(self.dw_mid, "compute_output_shape"):
            current_shape = self.dw_mid.compute_output_shape(current_shape)
        current_shape = self.pw_proj.compute_output_shape(current_shape)
        if hasattr(self.dw_end, "compute_output_shape"):
            current_shape = self.dw_end.compute_output_shape(current_shape)
        return current_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "dw_kernel_size_start": self.dw_kernel_size_start,
                "dw_kernel_size_mid": self.dw_kernel_size_mid,
                "dw_kernel_size_end": self.dw_kernel_size_end,
                "stride": self.stride,
                "dilation": self.dilation,
                "pad_type": self.pad_type,
                "noskip": self.noskip,
                "exp_ratio": self.exp_ratio,
                "act_layer": self.act_layer,
                "norm_layer": self.norm_layer,
                "se_layer": keras.saving.serialize_keras_object(self.se_layer),
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["se_layer"] = keras.saving.deserialize_keras_object(
            config.pop("se_layer")
        )
        return cls(**config)


class EdgeResidual(keras.layers.Layer):
    """Edge Residual block.

    This block is designed for efficiency on edge devices. It is a variant of
    the inverted residual block that uses a single expansion convolution.

    Args:
        filters: int. The number of output channels.
        exp_kernel_size: int. The kernel size for the expansion convolution.
        stride: int. The stride for the block.
        dilation: int. The dilation rate for convolutions.
        group_size: int. The group size for grouped convolutions.
        pad_type: str. The padding type for convolutions.
        expansion_in_chs: int. If greater than 0, forces the number of input
            channels for the expansion.
        noskip: bool. If `True`, the skip connection is disabled.
        exp_ratio: float. The expansion ratio for the middle channels.
        pw_kernel_size: int. The kernel size for the pointwise convolution.
        act_layer: str. The activation function to use.
        norm_layer: str. The normalization layer to use.
        se_layer: keras.layers.Layer. The Squeeze-and-Excitation layer to use.
        drop_path_rate: float. The stochastic depth rate.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        filters,
        exp_kernel_size=3,
        stride=1,
        dilation=1,
        group_size=0,
        pad_type="same",
        expansion_in_chs=0,
        noskip=False,
        exp_ratio=1.0,
        pw_kernel_size=1,
        act_layer="relu",
        norm_layer="batch_norm",
        se_layer=None,
        drop_path_rate=0.0,
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.exp_kernel_size = exp_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.group_size = group_size
        self.pad_type = pad_type
        self.expansion_in_chs = expansion_in_chs
        self.noskip = noskip
        self.exp_ratio = exp_ratio
        self.pw_kernel_size = pw_kernel_size
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = se_layer
        self.drop_path_rate = drop_path_rate
        self.data_format = data_format
        self.channel_axis = channel_axis

    def build(self, input_shape):
        super().build(input_shape)
        in_chs = input_shape[self.channel_axis]
        self.has_skip = (
            in_chs == self.filters and self.stride == 1
        ) and not self.noskip
        if self.expansion_in_chs > 0:
            mid_chs = adjust_channels(self.expansion_in_chs * self.exp_ratio)
        else:
            mid_chs = adjust_channels(in_chs * self.exp_ratio)
        groups = num_groups(self.group_size, mid_chs)
        use_bias = self.norm_layer == "rms_norm"
        self.conv_exp = ConvNormAct(
            mid_chs,
            self.exp_kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            pad_type=self.pad_type,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            bias=use_bias,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            dtype=self.dtype_policy,
        )
        self.se = (
            self.se_layer(
                filters=mid_chs,
                bottleneck_filters=adjust_channels(mid_chs * 0.25),
                squeeze_activation=self.act_layer,
                excite_activation="sigmoid",
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
            if self.se_layer
            else (lambda x, training=False: x)
        )
        self.conv_pwl = ConvNormAct(
            self.filters,
            self.pw_kernel_size,
            pad_type=self.pad_type,
            apply_act=False,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            bias=use_bias,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            dtype=self.dtype_policy,
        )
        self.drop_path = (
            DropPath(self.drop_path_rate, dtype=self.dtype_policy)
            if self.drop_path_rate > 0.0
            else (lambda x, training=False: x)
        )
        self.conv_exp.build(input_shape)
        conv_exp_output_shape = self.conv_exp.compute_output_shape(input_shape)
        if hasattr(self.se, "build"):
            self.se.build(conv_exp_output_shape)
        self.conv_pwl.build(conv_exp_output_shape)

    def call(self, x, training=False):
        shortcut = x
        x = self.conv_exp(x, training=training)
        x = self.se(x, training=training)
        x = self.conv_pwl(x, training=training)
        if self.has_skip:
            x = self.drop_path(x, training=training) + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "exp_kernel_size": self.exp_kernel_size,
                "stride": self.stride,
                "dilation": self.dilation,
                "group_size": self.group_size,
                "pad_type": self.pad_type,
                "expansion_in_chs": self.expansion_in_chs,
                "noskip": self.noskip,
                "exp_ratio": self.exp_ratio,
                "pw_kernel_size": self.pw_kernel_size,
                "act_layer": self.act_layer,
                "norm_layer": self.norm_layer,
                "se_layer": keras.saving.serialize_keras_object(self.se_layer),
                "drop_path_rate": self.drop_path_rate,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["se_layer"] = keras.saving.deserialize_keras_object(
            config.pop("se_layer")
        )
        return cls(**config)


class CondConvResidual(keras.layers.Layer):
    """Conditionally Parameterized Convolutional Residual block.

    This block uses a routing function to dynamically select and combine
    different convolutional experts based on the input.

    Args:
        filters: int. The number of output channels.
        dw_kernel_size: int. The kernel size for the depthwise convolution.
        stride: int. The stride for the block.
        dilation: int. The dilation rate for convolutions.
        pad_type: str. The padding type for convolutions.
        noskip: bool. If `True`, the skip connection is disabled.
        exp_ratio: float. The expansion ratio for the middle channels.
        exp_kernel_size: int. The kernel size for the expansion convolution.
        pw_kernel_size: int. The kernel size for the pointwise convolution.
        act_layer: str. The activation function to use.
        se_layer: keras.layers.Layer. The Squeeze-and-Excitation layer to use.
        num_experts: int. The number of experts to use.
        drop_path_rate: float. The stochastic depth rate.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        filters,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="same",
        noskip=False,
        exp_ratio=1.0,
        exp_kernel_size=1,
        pw_kernel_size=1,
        act_layer="relu",
        se_layer=None,
        num_experts=0,
        drop_path_rate=0.0,
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_type = pad_type
        self.noskip = noskip
        self.exp_ratio = exp_ratio
        self.exp_kernel_size = exp_kernel_size
        self.pw_kernel_size = pw_kernel_size
        self.act_layer = act_layer
        self.se_layer = se_layer
        self.num_experts = num_experts
        self.drop_path_rate = drop_path_rate
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.conv_kernel_initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="untruncated_normal"
        )
        self.dense_kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="uniform"
        )
        self.bias_initializer = "zeros"

    def build(self, input_shape):
        super().build(input_shape)
        in_chs = input_shape[self.channel_axis]
        self.has_skip = (
            in_chs == self.filters and self.stride == 1
        ) and not self.noskip
        mid_chs = adjust_channels(in_chs * self.exp_ratio)
        self.routing_fn = keras.layers.Dense(
            self.num_experts,
            dtype=self.dtype_policy,
            kernel_initializer=self.dense_kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.pool = keras.layers.GlobalAveragePooling2D(
            data_format=self.data_format, dtype=self.dtype_policy
        )
        self.conv_pw_experts = [
            keras.layers.Conv2D(
                filters=mid_chs,
                kernel_size=self.exp_kernel_size,
                padding=self.pad_type,
                use_bias=True,
                data_format=self.data_format,
                name=f"conv_pw_expert_{i}",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
            for i in range(self.num_experts)
        ]
        self.conv_dw_experts = [
            keras.layers.DepthwiseConv2D(
                kernel_size=self.dw_kernel_size,
                strides=self.stride,
                padding=self.pad_type,
                dilation_rate=self.dilation,
                use_bias=True,
                data_format=self.data_format,
                name=f"conv_dw_expert_{i}",
                depthwise_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
            for i in range(self.num_experts)
        ]
        self.conv_pwl_experts = [
            keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.pw_kernel_size,
                padding=self.pad_type,
                use_bias=True,
                data_format=self.data_format,
                name=f"conv_pwl_expert_{i}",
                kernel_initializer=self.conv_kernel_initializer,
                bias_initializer=self.bias_initializer,
                dtype=self.dtype_policy,
            )
            for i in range(self.num_experts)
        ]
        self.bn1 = keras.layers.BatchNormalization(
            axis=self.channel_axis,
            dtype=self.dtype_policy,
            gamma_initializer="ones",
            beta_initializer="zeros",
        )
        self.act1 = keras.layers.Activation(
            self.act_layer, dtype=self.dtype_policy
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=self.channel_axis,
            dtype=self.dtype_policy,
            gamma_initializer="ones",
            beta_initializer="zeros",
        )
        self.act2 = keras.layers.Activation(
            self.act_layer, dtype=self.dtype_policy
        )
        self.bn3 = keras.layers.BatchNormalization(
            axis=self.channel_axis,
            dtype=self.dtype_policy,
            gamma_initializer="ones",
            beta_initializer="zeros",
        )
        self.se = (
            self.se_layer(
                filters=mid_chs,
                bottleneck_filters=adjust_channels(mid_chs * 0.25),
                squeeze_activation=self.act_layer,
                excite_activation="sigmoid",
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
            if self.se_layer
            else (lambda x, training=False: x)
        )
        self.drop_path = (
            DropPath(self.drop_path_rate, dtype=self.dtype_policy)
            if self.drop_path_rate > 0.0
            else (lambda x, training=False: x)
        )
        pooled_shape = self.pool.compute_output_shape(input_shape)
        self.routing_fn.build(pooled_shape)
        for expert in self.conv_pw_experts:
            expert.build(input_shape)
        pw_out_shape = self.conv_pw_experts[0].compute_output_shape(input_shape)
        self.bn1.build(pw_out_shape)
        for expert in self.conv_dw_experts:
            expert.build(pw_out_shape)
        dw_out_shape = self.conv_dw_experts[0].compute_output_shape(
            pw_out_shape
        )
        self.bn2.build(dw_out_shape)
        if hasattr(self.se, "build"):
            self.se.build(dw_out_shape)
        for expert in self.conv_pwl_experts:
            expert.build(dw_out_shape)
        pwl_out_shape = self.conv_pwl_experts[0].compute_output_shape(
            dw_out_shape
        )
        self.bn3.build(pwl_out_shape)

    def _apply_cond_conv(self, x, experts, routing_weights):
        outputs = []
        for i, expert in enumerate(experts):
            expert_out = expert(x)
            weight = keras.ops.reshape(routing_weights[:, i], (-1, 1, 1, 1))
            outputs.append(expert_out * weight)
        return keras.ops.sum(outputs, axis=0)

    def call(self, x, training=False):
        shortcut = x
        pooled_inputs = self.pool(x)
        routing_weights = keras.activations.sigmoid(
            self.routing_fn(pooled_inputs)
        )
        x = self._apply_cond_conv(x, self.conv_pw_experts, routing_weights)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self._apply_cond_conv(x, self.conv_dw_experts, routing_weights)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.se(x, training=training)
        x = self._apply_cond_conv(x, self.conv_pwl_experts, routing_weights)
        x = self.bn3(x, training=training)
        if self.has_skip:
            x = self.drop_path(x, training=training) + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "dw_kernel_size": self.dw_kernel_size,
                "stride": self.stride,
                "dilation": self.dilation,
                "pad_type": self.pad_type,
                "noskip": self.noskip,
                "exp_ratio": self.exp_ratio,
                "exp_kernel_size": self.exp_kernel_size,
                "pw_kernel_size": self.pw_kernel_size,
                "act_layer": self.act_layer,
                "se_layer": keras.saving.serialize_keras_object(self.se_layer),
                "num_experts": self.num_experts,
                "drop_path_rate": self.drop_path_rate,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["se_layer"] = keras.saving.deserialize_keras_object(
            config.pop("se_layer")
        )
        return cls(**config)


class MobileNetV5MultiScaleFusionAdapter(keras.layers.Layer):
    """Multi-Scale Fusion Adapter for MobileNetV5.

    This layer fuses feature maps from different scales of the backbone,
    concatenates them, processes them through a FFN (Feed-Forward Network),
    and then resizes the output to a target resolution.

    Args:
        in_chs: list of int. A list of channel counts for each input feature
            map.
        filters: int. The number of output channels.
        output_resolution: int or tuple. The target output resolution.
        expansion_ratio: float. The expansion ratio for the FFN.
        interpolation_mode: str. The interpolation mode for upsampling feature
            maps.
        layer_scale_init_value: float. The initial value for layer scale. If
            `None`, layer scale is not used.
        noskip: bool. If `True`, the skip connection in the FFN is disabled.
        act_layer: str. The activation function to use.
        norm_layer: str. The normalization layer to use.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        in_chs,
        filters,
        output_resolution,
        expansion_ratio=2.0,
        interpolation_mode="nearest",
        layer_scale_init_value=None,
        noskip=True,
        act_layer="gelu",
        norm_layer="rms_norm",
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_chs = in_chs
        self.filters = filters
        self.output_resolution_arg = output_resolution
        self.expansion_ratio = expansion_ratio
        self.interpolation_mode = interpolation_mode
        self.layer_scale_init_value = layer_scale_init_value
        self.noskip = noskip
        self.act_layer = act_layer
        self.norm_layer_name = norm_layer
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.in_channels = sum(in_chs)
        if isinstance(output_resolution, int):
            self.output_resolution = (output_resolution, output_resolution)
        else:
            self.output_resolution = output_resolution
        self.ffn = UniversalInvertedResidual(
            filters=self.filters,
            dw_kernel_size_mid=0,
            exp_ratio=expansion_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            noskip=noskip,
            layer_scale_init_value=layer_scale_init_value,
            data_format=self.data_format,
            channel_axis=self.channel_axis,
            dtype=self.dtype_policy,
        )
        if norm_layer == "rms_norm":
            self.norm = RmsNorm2d(
                self.filters,
                data_format=self.data_format,
                gamma_initializer="ones",
                channel_axis=self.channel_axis,
                dtype=self.dtype_policy,
            )
        else:
            self.norm = keras.layers.BatchNormalization(
                axis=self.channel_axis,
                gamma_initializer="ones",
                beta_initializer="zeros",
                dtype=self.dtype_policy,
            )

    def build(self, input_shape):
        super().build(input_shape)
        ffn_input_shape = list(input_shape[0])
        if self.data_format == "channels_first":
            ffn_input_shape[1] = self.in_channels
        else:
            ffn_input_shape[-1] = self.in_channels
        self.ffn.build(tuple(ffn_input_shape))
        norm_input_shape = self.ffn.compute_output_shape(tuple(ffn_input_shape))
        self.norm.build(norm_input_shape)

    def call(self, inputs, training=False):
        shape_hr = keras.ops.shape(inputs[0])
        if self.data_format == "channels_first":
            high_resolution = (shape_hr[2], shape_hr[3])
        else:
            high_resolution = (shape_hr[1], shape_hr[2])
        resized_inputs = []
        for img in inputs:
            if self.data_format == "channels_first":
                img_transposed = keras.ops.transpose(img, (0, 2, 3, 1))
            else:
                img_transposed = img
            img_resized = keras.ops.image.resize(
                img_transposed,
                size=high_resolution,
                interpolation=self.interpolation_mode,
            )
            if self.data_format == "channels_first":
                resized_inputs.append(
                    keras.ops.transpose(img_resized, (0, 3, 1, 2))
                )
            else:
                resized_inputs.append(img_resized)
        channel_cat_imgs = keras.ops.concatenate(
            resized_inputs, axis=self.channel_axis
        )
        img = self.ffn(channel_cat_imgs, training=training)
        if (
            high_resolution[0] != self.output_resolution[0]
            or high_resolution[1] != self.output_resolution[1]
        ):
            h_in, w_in = high_resolution
            h_out, w_out = self.output_resolution
            if h_in % h_out == 0 and w_in % w_out == 0:
                h_stride = h_in // h_out
                w_stride = w_in // w_out
                img = keras.ops.nn.average_pool(
                    img,
                    pool_size=(h_stride, w_stride),
                    strides=(h_stride, w_stride),
                    padding="valid",
                    data_format=self.data_format,
                )
            else:
                if self.data_format == "channels_first":
                    img_transposed = keras.ops.transpose(img, (0, 2, 3, 1))
                else:
                    img_transposed = img
                img_resized = keras.ops.image.resize(
                    img_transposed,
                    size=self.output_resolution,
                    interpolation="bilinear",
                )
                if self.data_format == "channels_first":
                    img = keras.ops.transpose(img_resized, (0, 3, 1, 2))
                else:
                    img = img_resized
        img = self.norm(img, training=training)
        return img

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        if self.data_format == "channels_first":
            return (
                batch_size,
                self.filters,
                self.output_resolution[0],
                self.output_resolution[1],
            )
        else:
            return (
                batch_size,
                self.output_resolution[0],
                self.output_resolution[1],
                self.filters,
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chs": self.in_chs,
                "filters": self.filters,
                "output_resolution": self.output_resolution_arg,
                "expansion_ratio": self.expansion_ratio,
                "interpolation_mode": self.interpolation_mode,
                "layer_scale_init_value": self.layer_scale_init_value,
                "noskip": self.noskip,
                "act_layer": self.act_layer,
                "norm_layer": self.norm_layer_name,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config
