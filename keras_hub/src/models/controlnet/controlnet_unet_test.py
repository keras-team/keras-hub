import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_unet import ControlNetUNet


def test_controlnet_unet_smoke():
    model = ControlNetUNet()

    image = tf.random.uniform((1, 128, 128, 3))
    control_features = {
        "scale_1": tf.random.uniform((1, 128, 128, 64))
    }

    outputs = model(image, control_features)

    assert outputs.shape == (1, 128, 128, 3)
