import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_preprocessor import ControlNetPreprocessor


def test_controlnet_preprocessor_output_shape():
    layer = ControlNetPreprocessor(target_size=(128, 128))

    x = tf.random.uniform((1, 256, 256, 3), maxval=255, dtype=tf.float32)
    y = layer(x)

    assert y.shape == (1, 128, 128, 3)


def test_controlnet_preprocessor_scaling():
    layer = ControlNetPreprocessor(target_size=(64, 64))

    x = tf.ones((1, 128, 128, 3)) * 255.0
    y = layer(x)

    assert tf.reduce_max(y).numpy() <= 1.0
    assert tf.reduce_min(y).numpy() >= 0.0


def test_controlnet_preprocessor_dtype():
    layer = ControlNetPreprocessor(target_size=(64, 64))

    x = tf.random.uniform((1, 128, 128, 3), maxval=255, dtype=tf.float32)
    y = layer(x)

    assert y.dtype == tf.float32
