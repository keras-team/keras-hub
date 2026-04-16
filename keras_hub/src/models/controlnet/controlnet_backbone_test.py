import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_backbone import (
    ControlNetBackbone,
)


def test_controlnet_backbone_smoke():
    model = ControlNetBackbone()
    x = tf.random.uniform((1, 512, 512, 1))
    outputs = model(x)
    assert isinstance(outputs, dict)


def test_controlnet_backbone_required_keys():
    model = ControlNetBackbone()
    x = tf.random.uniform((1, 512, 512, 1))
    outputs = model(x)

    assert "scale_1" in outputs
    assert "scale_2" in outputs
    assert "scale_3" in outputs


def test_controlnet_backbone_rank():
    model = ControlNetBackbone()
    x = tf.random.uniform((2, 256, 256, 1))
    outputs = model(x)

    for v in outputs.values():
        assert len(v.shape) == 4
        assert v.shape[0] == 2


def test_controlnet_backbone_spatial_scaling():
    model = ControlNetBackbone()
    x = tf.random.uniform((1, 256, 256, 1))
    outputs = model(x)

    assert outputs["scale_1"].shape[1:3] == (256, 256)
    assert outputs["scale_2"].shape[1:3] == (128, 128)
    assert outputs["scale_3"].shape[1:3] == (64, 64)
