"""NumPy helpers shared by the pure Python paths of audio converters.

`keras_hub.layers.AudioConverter` subclasses historically computed
spectrograms with `tf.signal`. These helpers mirror the `tf.signal` semantics
with NumPy so audio preprocessing can run inside Grain worker processes (and
on Jax/Torch setups without TensorFlow installed).
"""

import numpy as np


def hann_window(length, periodic=True, dtype="float32"):
    """Return a Hann window matching `tf.signal.hann_window`.

    Args:
        length: int. The length of the window.
        periodic: bool. If `True` (the default, and the `tf.signal` default),
            generate a periodic window for use with spectral analysis. If
            `False`, generate a symmetric window.
        dtype: str. The dtype of the returned window.
    """
    if length <= 1:
        return np.ones(max(length, 0), dtype=dtype)
    denominator = length if periodic else length - 1
    arange = np.arange(length, dtype="float64")
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * arange / denominator)
    return window.astype(dtype)


def frame_signal(x, frame_length, frame_step):
    """Slice `x` into overlapping frames, matching `tf.signal.frame`.

    Framing happens along the last axis with `pad_end=False`, so incomplete
    trailing frames are dropped.

    Args:
        x: NumPy array of shape `(..., samples)`.
        frame_length: int. The size of each frame.
        frame_step: int. The hop size between consecutive frames.

    Returns:
        A NumPy array of shape `(..., num_frames, frame_length)`.
    """
    num_samples = x.shape[-1]
    if num_samples < frame_length:
        num_frames = 0
    else:
        num_frames = 1 + (num_samples - frame_length) // frame_step
    if num_frames <= 0:
        return np.zeros(x.shape[:-1] + (0, frame_length), dtype=x.dtype)
    starts = np.arange(num_frames) * frame_step
    indices = starts[:, None] + np.arange(frame_length)[None, :]
    return x[..., indices]
