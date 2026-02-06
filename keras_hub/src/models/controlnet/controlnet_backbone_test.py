import tensorflow as tf
from keras_hub.src.models.controlnet.controlnet_backbone import (
    ControlNetBackbone,
)


def test_controlnet_backbone_smoke():
    """Basic smoke test: model builds and runs."""
    model = ControlNetBackbone()

    x = tf.random.uniform((1, 512, 512, 1))
    outputs = model(x)

    assert isinstance(outputs, dict)


def test_controlnet_backbone_required_keys():
    """Ensure expected feature scales exist."""
    model = ControlNetBackbone()
    x = tf.random.uniform((1, 512, 512, 1))

    outputs = model(x)

    assert "scale_1" in outputs
    assert "scale_2" in outputs
    assert "scale_3" in outputs


def test_controlnet_backbone_rank():
    """Each output should be a 4D tensor (B, H, W, C)."""
    model = ControlNetBackbone()
    x = tf.random.uniform((2, 256, 256, 1))

    outputs = model(x)

    for v in outputs.values():
        assert len(v.shape) == 4
        assert v.shape[0] == 2  
