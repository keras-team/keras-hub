import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet import ControlNet


def test_controlnet_full_model_smoke():
    model = ControlNet()

    inputs = {
        "image": tf.random.uniform((1, 128, 128, 3)),
        "control": tf.random.uniform((1, 128, 128, 3)),
    }

    outputs = model(inputs)

    assert outputs.shape == (1, 128, 128, 3)
