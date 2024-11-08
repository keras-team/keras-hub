from keras import ops
from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D


def apply_conv_bn(x, num_channels, kernel_size=1, strides=1,
                  activation="swish", name="conv_bn"):
    if kernel_size > 1:
        x = ZeroPadding2D(kernel_size // 2, name=f"{name}_pad")(x)
    conv_kwargs = {"use_bias": False, "name": f"{name}_conv"}
    x = Conv2D(num_channels, kernel_size, strides, "valid", **conv_kwargs)(x)
    x = BatchNormalization(momentum=0.97, epsilon=1e-3, name=f"{name}_bn")(x)
    x = Activation(activation, name=name)(x)
    return x


def compute_hidden_channels(channels, expansion):
    return int(channels * expansion)


def get_default_channels(channels, x):
    return channels if channels > 0 else x.shape[-1]


def compute_short_and_deep(x, hidden_channels, activation, name):
    x = apply_conv_bn(x, 2 * hidden_channels, 1, 1, activation, f"{name}_pre")
    short, deep = ops.split(x, 2, axis=-1)
    return short, deep


def apply_conv_block(y, channels, activation, shortcut, name):
    x = apply_conv_bn(y, channels, 3, 1, activation, f"{name}_1")
    x = apply_conv_bn(x, channels, 3, 1, activation, f"{name}_2")
    x = (y + x) if shortcut else x
    return x


def apply_CSP(x, channels=-1, depth=2, shortcut=True, expansion=0.5,
              activation="swish", name="csp_block"):
    channels = get_default_channels(channels, x)
    hidden_channels = compute_hidden_channels(channels, expansion)
    short, deep = compute_short_and_deep(x, hidden_channels, activation, name)
    out = [short, deep]
    conv_args = (hidden_channels, activation, shortcut)
    for depth_arg in range(depth):
        deep = apply_conv_block(deep, *conv_args, f"{name}_pre_{depth_arg}")
        out.append(deep)
    out = ops.concatenate(out, axis=-1)
    out = apply_conv_bn(out, channels, 1, 1, activation, f"{name}_output")
    return out
