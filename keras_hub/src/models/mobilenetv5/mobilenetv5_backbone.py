import re
from copy import deepcopy

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobilenet.mobilenet_backbone import SqueezeAndExcite2D
from keras_hub.src.models.mobilenet.util import adjust_channels


class MobileNetV5RMSNormalization(keras.layers.Layer):
    """Root Mean Square Normalization layer."""

    def __init__(self, epsilon=1e-6, axis=-1, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon
        self.axis = axis

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            shape = (input_shape[-1],)
        else:
            shape = [1] * len(input_shape)
            shape[self.axis] = input_shape[self.axis]
            shape = tuple(shape)

        self.scale = self.add_weight(
            name="scale",
            shape=shape,
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        norm = keras.ops.square(inputs)
        mean_norm = keras.ops.mean(norm, axis=self.axis, keepdims=True)
        normalized_inputs = inputs * keras.ops.rsqrt(mean_norm + self.epsilon)
        return normalized_inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon, "axis": self.axis})
        return config


class LayerScale(keras.layers.Layer):
    """LayerScale layer."""

    def __init__(self, init_values=1e-5, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.init_values = init_values

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_values": self.init_values})
        return config


class ConvNormAct(keras.layers.Layer):
    """Convolution, Normalization, and Activation block."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        dilation_rate=1,
        groups=1,
        use_bias=False,
        layer_norm_epsilon=1e-6,
        act_layer="gelu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.use_bias = use_bias
        self.act_layer = act_layer
        self.layer_norm_epsilon = layer_norm_epsilon

        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            use_bias=self.use_bias,
            name=f"{self.name}_conv",
            dtype=dtype,
        )
        self.norm = MobileNetV5RMSNormalization(
            name=f"{self.name}_norm",
            epsilon=self.layer_norm_epsilon,
            dtype=dtype,
        )
        self.act = (
            keras.layers.Activation(
                self.act_layer, name=f"{self.name}_act", dtype=dtype
            )
            if self.act_layer
            else keras.layers.Identity(dtype=dtype)
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.act(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "use_bias": self.use_bias,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "act_layer": self.act_layer,
            }
        )
        return config


class UniversalInvertedResidualBlock(keras.layers.Layer):
    """Universal Inverted Residual Block (UIB)."""

    def __init__(
        self,
        in_chs,
        out_chs,
        exp_ratio,
        dw_kernel_size_mid,
        dw_kernel_size_start=0,
        dw_kernel_size_end=0,
        stride=1,
        se_ratio=None,
        noskip=False,
        act_layer="gelu",
        layer_norm_epsilon=1e-6,
        layer_scale_init_value=1e-5,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.exp_ratio = exp_ratio
        self.dw_kernel_size_mid = dw_kernel_size_mid
        self.dw_kernel_size_start = dw_kernel_size_start
        self.dw_kernel_size_end = dw_kernel_size_end
        self.stride = stride
        self.se_ratio = se_ratio
        self.noskip = noskip
        self.act_layer = act_layer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_scale_init_value = layer_scale_init_value
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        mid_chs = adjust_channels(self.in_chs * self.exp_ratio)

        # DW Conv Start
        if self.dw_kernel_size_start > 0:
            self.dw_start = ConvNormAct(
                self.in_chs,
                self.dw_kernel_size_start,
                strides=self.stride if not self.dw_kernel_size_mid else 1,
                padding="same",
                groups=self.in_chs,
                layer_norm_epsilon=self.layer_norm_epsilon,
                act_layer=None,
                name=f"{self.name}_dw_start",
                dtype=dtype,
            )
        else:
            self.dw_start = keras.layers.Identity(dtype=dtype)

        # PW Expansion
        self.pw_exp = ConvNormAct(
            mid_chs,
            1,
            layer_norm_epsilon=self.layer_norm_epsilon,
            act_layer=self.act_layer,
            name=f"{self.name}_pw_exp",
            dtype=dtype,
        )

        # DW Conv Middle
        if self.dw_kernel_size_mid > 0:
            self.dw_mid = ConvNormAct(
                mid_chs,
                self.dw_kernel_size_mid,
                strides=self.stride,
                padding="same",
                groups=mid_chs,
                layer_norm_epsilon=self.layer_norm_epsilon,
                act_layer=self.act_layer,
                name=f"{self.name}_dw_mid",
                dtype=dtype,
            )
        else:
            self.dw_mid = keras.layers.Identity(dtype=dtype)

        # Squeeze and Excite
        if self.se_ratio and self.se_ratio > 0:
            self.se = SqueezeAndExcite2D(
                filters=mid_chs,
                bottleneck_filters=adjust_channels(mid_chs * self.se_ratio),
                squeeze_activation=self.act_layer,
                excite_activation="sigmoid",
                name=f"{self.name}_se",
                dtype=dtype,
            )
        else:
            self.se = keras.layers.Identity(dtype=dtype)

        # PW Projection
        self.pw_proj = ConvNormAct(
            self.out_chs,
            1,
            layer_norm_epsilon=self.layer_norm_epsilon,
            act_layer=None,
            name=f"{self.name}_pw_proj",
            dtype=dtype,
        )

        # DW Conv End
        if self.dw_kernel_size_end > 0:
            self.dw_end = ConvNormAct(
                self.out_chs,
                self.dw_kernel_size_end,
                strides=self.stride
                if not self.dw_kernel_size_start and not self.dw_kernel_size_mid
                else 1,
                padding="same",
                groups=self.out_chs,
                layer_norm_epsilon=self.layer_norm_epsilon,
                act_layer=None,
                name=f"{self.name}_dw_end",
                dtype=dtype,
            )
        else:
            self.dw_end = keras.layers.Identity(dtype=dtype)

        # Layer Scale
        if self.layer_scale_init_value:
            self.layer_scale = LayerScale(
                self.layer_scale_init_value, name=f"{self.name}_ls", dtype=dtype
            )
        else:
            self.layer_scale = keras.layers.Identity(dtype=dtype)

    def call(self, inputs):
        shortcut = inputs
        x = self.dw_start(inputs)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x

    def compute_output_shape(self, input_shape):
        current_shape = input_shape
        if hasattr(self, "dw_start") and not isinstance(
            self.dw_start, keras.layers.Identity
        ):
            current_shape = self.dw_start.compute_output_shape(current_shape)
        current_shape = self.pw_exp.compute_output_shape(current_shape)
        if hasattr(self, "dw_mid") and not isinstance(
            self.dw_mid, keras.layers.Identity
        ):
            current_shape = self.dw_mid.compute_output_shape(current_shape)
        current_shape = self.pw_proj.compute_output_shape(current_shape)
        if hasattr(self, "dw_end") and not isinstance(
            self.dw_end, keras.layers.Identity
        ):
            current_shape = self.dw_end.compute_output_shape(current_shape)
        return current_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chs": self.in_chs,
                "out_chs": self.out_chs,
                "exp_ratio": self.exp_ratio,
                "dw_kernel_size_mid": self.dw_kernel_size_mid,
                "dw_kernel_size_start": self.dw_kernel_size_start,
                "dw_kernel_size_end": self.dw_kernel_size_end,
                "stride": self.stride,
                "se_ratio": self.se_ratio,
                "noskip": self.noskip,
                "act_layer": self.act_layer,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config


class MobileAttentionBlock(keras.layers.Layer):
    """Mobile Attention Block using Multi-Query Attention."""

    def __init__(
        self,
        in_chs,
        out_chs,
        num_heads,
        key_dim,
        value_dim,
        kv_stride=1,
        dw_kernel_size=3,
        noskip=False,
        stride=1,
        layer_norm_epsilon=1e-6,
        layer_scale_init_value=1e-5,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kv_stride = kv_stride
        self.dw_kernel_size = dw_kernel_size
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_scale_init_value = layer_scale_init_value

        self.norm = MobileNetV5RMSNormalization(
            name=f"{self.name}_norm",
            epsilon=self.layer_norm_epsilon,
            dtype=dtype,
        )
        self.k_depth = self.key_dim
        self.v_depth = self.value_dim

        self.q_proj = keras.layers.Conv2D(
            filters=self.num_heads * self.key_dim,
            kernel_size=1,
            use_bias=False,
            name=f"{self.name}_q_proj",
            dtype=dtype,
        )
        self.k_proj = keras.layers.Conv2D(
            filters=self.k_depth,
            kernel_size=self.dw_kernel_size,
            strides=self.kv_stride,
            padding="same",
            groups=self.k_depth,
            use_bias=False,
            name=f"{self.name}_k_proj",
            dtype=dtype,
        )
        self.v_proj = keras.layers.Conv2D(
            filters=self.v_depth,
            kernel_size=self.dw_kernel_size,
            strides=self.kv_stride,
            padding="same",
            groups=self.v_depth,
            use_bias=False,
            name=f"{self.name}_v_proj",
            dtype=dtype,
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.v_depth,
            output_shape=self.out_chs,
            use_bias=True,
            name=f"{self.name}_attn",
            dtype=dtype,
        )
        if self.layer_scale_init_value:
            self.layer_scale = LayerScale(
                self.layer_scale_init_value, name=f"{self.name}_ls", dtype=dtype
            )
        else:
            self.layer_scale = keras.layers.Identity(dtype=dtype)

    def _to_seq(self, x):
        h, w, c = (
            keras.ops.shape(x)[1],
            keras.ops.shape(x)[2],
            keras.ops.shape(x)[3],
        )
        return keras.ops.reshape(x, (-1, h * w, c))

    def _to_img(self, x, h, w):
        b, _, c = keras.ops.shape(x)
        return keras.ops.reshape(x, (b, h, w, c))

    def call(self, inputs):
        shortcut = inputs
        x = self.norm(inputs)

        h, w = keras.ops.shape(x)[1], keras.ops.shape(x)[2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_seq = self._to_seq(q)
        k_seq = self._to_seq(k)
        v_seq = self._to_seq(v)

        # Multi-Query is achieved by repeating K and V for each head.
        k_seq_multi = keras.ops.repeat(k_seq, self.num_heads, axis=-1)
        v_seq_multi = keras.ops.repeat(v_seq, self.num_heads, axis=-1)

        attn_output = self.attention(
            query=q_seq, key=k_seq_multi, value=v_seq_multi
        )
        attn_output = self._to_img(attn_output, h, w)

        attn_output = self.layer_scale(attn_output)

        if self.has_skip:
            attn_output = attn_output + shortcut

        return attn_output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_chs
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chs": self.in_chs,
                "out_chs": self.out_chs,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "kv_stride": self.kv_stride,
                "dw_kernel_size": self.dw_kernel_size,
                "noskip": self.noskip,
                "stride": self.stride,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config


class MobileNetV5MultiScaleFusionAdapter(keras.layers.Layer):
    """Multi-layer fusion token adapter for MobileNetV5."""

    def __init__(
        self,
        in_chs_list,
        out_chs,
        output_resolution,
        expansion_ratio=2.0,
        interpolation_mode="bilinear",
        layer_scale_init_value=1e-5,
        noskip=True,
        act_layer="gelu",
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_chs_list = in_chs_list
        self.in_channels = sum(in_chs_list)
        self.out_chs = out_chs
        self.output_resolution = (output_resolution, output_resolution)
        self.expansion_ratio = expansion_ratio
        self.interpolation_mode = interpolation_mode
        self.layer_scale_init_value = layer_scale_init_value
        self.noskip = noskip
        self.act_layer = act_layer
        self.layer_norm_epsilon = layer_norm_epsilon

        self.ffn = UniversalInvertedResidualBlock(
            in_chs=self.in_channels,
            out_chs=self.out_chs,
            dw_kernel_size_mid=0,
            exp_ratio=self.expansion_ratio,
            act_layer=self.act_layer,
            layer_norm_epsilon=self.layer_norm_epsilon,
            noskip=self.noskip,
            layer_scale_init_value=self.layer_scale_init_value,
            name=f"{self.name}_ffn",
            dtype=dtype,
        )
        self.norm = MobileNetV5RMSNormalization(
            name=f"{self.name}_norm",
            epsilon=self.layer_norm_epsilon,
            dtype=dtype,
        )

    def call(self, inputs):
        # inputs is a list of tensors [B, H, W, C]
        high_res_shape = keras.ops.shape(inputs[0])
        high_resolution = (high_res_shape[1], high_res_shape[2])

        resized_inputs = []
        for img in inputs:
            img_shape = keras.ops.shape(img)
            if (
                img_shape[1] < high_resolution[0]
                or img_shape[2] < high_resolution[1]
            ):
                img = keras.layers.UpSampling2D(
                    size=(
                        high_resolution[0] // img_shape[1],
                        high_resolution[1] // img_shape[2],
                    ),
                    interpolation=self.interpolation_mode,
                    dtype=self.dtype,
                )(img)
            resized_inputs.append(img)

        x = keras.layers.Concatenate(axis=-1, dtype=self.dtype)(resized_inputs)
        x = self.ffn(x)

        x_shape = keras.ops.shape(x)
        if (
            x_shape[1] != self.output_resolution[0]
            or x_shape[2] != self.output_resolution[1]
        ):
            h_strides = x_shape[1] // self.output_resolution[0]
            w_strides = x_shape[2] // self.output_resolution[1]
            x = keras.layers.AveragePooling2D(
                pool_size=(h_strides, w_strides),
                strides=(h_strides, w_strides),
            )(x)

        x = self.norm(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[1] = self.output_resolution[0]
        output_shape[2] = self.output_resolution[1]
        output_shape[-1] = self.out_chs
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chs_list": self.in_chs_list,
                "out_chs": self.out_chs,
                "output_resolution": self.output_resolution[0],
                "expansion_ratio": self.expansion_ratio,
                "interpolation_mode": self.interpolation_mode,
                "layer_scale_init_value": self.layer_scale_init_value,
                "noskip": self.noskip,
                "act_layer": self.act_layer,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


# Helper functions adapted from timm
def _decode_block_str(block_str):
    """Decode block definition string."""
    ops = block_str.split("_")
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    for op in ops:
        if op == "noskip":
            options["noskip"] = True
        elif op.startswith("n"):
            # activation fn
            v = op[1:]
            if v == "re":
                value = "relu"
            elif v == "hs":
                value = "hard_swish"
            elif v == "sw":
                value = "swish"
            elif v == "ge":
                value = "gelu"
            else:
                continue
            options["act_layer"] = value
        else:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    args = {
        "block_type": block_type,
        "out_chs": int(options["c"]),
        "stride": int(options.get("s", 1)),
        "act_layer": options.get("act_layer", "gelu"),
    }

    if block_type == "er":
        args.update(
            dict(
                exp_ratio=float(options["e"]),
                dw_kernel_size_mid=int(options["k"]),
            )
        )
    elif block_type == "uir":
        args.update(
            dict(
                dw_kernel_size_start=int(options.get("a", 0)),
                dw_kernel_size_mid=int(options["k"]),
                dw_kernel_size_end=int(options.get("p", 0)),
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
            )
        )
    elif block_type == "mqa":
        args.update(
            dict(
                dw_kernel_size=int(options["k"]),
                num_heads=int(options["h"]),
                key_dim=int(options["d"]),
                value_dim=int(options["d"]),
                kv_stride=int(options.get("v", 1)),
            )
        )
    else:
        raise ValueError(f"Unknown block type {block_type}")

    return args, int(options["r"])


def decode_arch_def(arch_def):
    """Decode architecture definition."""
    arch_args = []
    for stack_args_str in arch_def:
        stack_args = []
        for block_str in stack_args_str:
            args, repeats = _decode_block_str(block_str)
            stack_args.extend([deepcopy(args) for _ in range(repeats)])
        arch_args.append(stack_args)
    return arch_args


@keras_hub_export("keras_hub.models.MobileNetV5Backbone")
class MobileNetV5Backbone(Backbone):
    """Instantiates the MobileNetV5 architecture."""

    def __init__(
        self,
        block_args,
        stem_size=64,
        stem_bias=True,
        msfa_indices=(-2, -1),
        msfa_output_resolution=16,
        num_features=2048,
        image_shape=(None, None, 3),
        act_layer="gelu",
        layer_norm_epsilon=1e-6,
        layer_scale_init_value=1e-5,
        dtype=None,
        **kwargs,
    ):
        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )
        inputs = keras.layers.Input(shape=image_shape)

        # Stem
        x = ConvNormAct(
            filters=stem_size,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=stem_bias,
            layer_norm_epsilon=layer_norm_epsilon,
            act_layer=act_layer,
            name="stem",
            dtype=dtype,
        )(inputs)

        # Build blocks
        decoded_arch = decode_arch_def(block_args)

        feature_maps = [x]  # index 0 is stem output
        current_in_chs = stem_size

        for stack_idx, stack_args in enumerate(decoded_arch):
            for block_idx, b_args in enumerate(stack_args):
                block_name = f"stack{stack_idx}_block{block_idx}"
                block_type = b_args.pop("block_type")
                b_args["in_chs"] = current_in_chs
                b_args["out_chs"] = adjust_channels(b_args["out_chs"])

                if block_type == "er" or block_type == "uir":
                    block = UniversalInvertedResidualBlock(
                        **b_args,
                        layer_norm_epsilon=layer_norm_epsilon,
                        layer_scale_init_value=layer_scale_init_value,
                        name=block_name,
                        dtype=dtype,
                    )
                elif block_type == "mqa":
                    # Pop act_layer as MobileAttentionBlock doesn't use it directly
                    b_args.pop("act_layer", None)
                    block = MobileAttentionBlock(
                        **b_args,
                        layer_norm_epsilon=layer_norm_epsilon,
                        layer_scale_init_value=layer_scale_init_value,
                        name=block_name,
                        dtype=dtype,
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")

                x = block(x)
                current_in_chs = b_args["out_chs"]
            feature_maps.append(x)

        # Multi-Scale Fusion Adapter (MSFA)
        msfa_input_maps = [feature_maps[i] for i in msfa_indices]

        x = MobileNetV5MultiScaleFusionAdapter(
            in_chs_list=[m.shape[channel_axis] for m in msfa_input_maps],
            out_chs=num_features,
            output_resolution=msfa_output_resolution,
            layer_norm_epsilon=layer_norm_epsilon,
            act_layer=act_layer,
            name="msfa",
            dtype=dtype,
        )(msfa_input_maps)

        super().__init__(inputs=inputs, outputs=x, dtype=dtype, **kwargs)

        # Store config
        self.block_args = block_args
        self.stem_size = stem_size
        self.stem_bias = stem_bias
        self.msfa_indices = msfa_indices
        self.msfa_output_resolution = msfa_output_resolution
        self.num_features = num_features
        self.image_shape = image_shape
        self.act_layer = act_layer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_scale_init_value = layer_scale_init_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_args": self.block_args,
                "stem_size": self.stem_size,
                "stem_bias": self.stem_bias,
                "msfa_indices": self.msfa_indices,
                "msfa_output_resolution": self.msfa_output_resolution,
                "num_features": self.num_features,
                "image_shape": self.image_shape,
                "act_layer": self.act_layer,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config
