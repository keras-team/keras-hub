import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_presets import from_preset


def test_controlnet_from_preset():
    model = from_preset("controlnet_base")

    inputs = {
        "image": tf.random.uniform((1, 128, 128, 3)),
        "control": tf.random.uniform((1, 128, 128, 3)),
    }

    outputs = model(inputs)

    assert outputs.shape == (1, 128, 128, 3)
