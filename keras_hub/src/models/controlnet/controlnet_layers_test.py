import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_layers import ZeroConv2D


def test_zero_conv_output_shape():
    layer = ZeroConv2D(64)
    x = tf.random.uniform((1, 128, 128, 3))
    y = layer(x)
    assert y.shape == (1, 128, 128, 64)


def test_zero_conv_initial_output_is_zero():
    layer = ZeroConv2D(64)
    x = tf.random.uniform((1, 64, 64, 3))
    y = layer(x)
    assert tf.reduce_sum(tf.abs(y)).numpy() == 0.0
