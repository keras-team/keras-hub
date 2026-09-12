import numpy as np
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.audio_utils import frame_signal
from keras_hub.src.utils.audio_utils import hann_window


class AudioUtilsTest(TestCase):
    def test_hann_window_matches_tf(self):
        for length in (1, 2, 8, 400):
            self.assertAllClose(
                hann_window(length),
                tf.signal.hann_window(length, periodic=True),
            )
            self.assertAllClose(
                hann_window(length, periodic=False),
                tf.signal.hann_window(length, periodic=False),
            )

    def test_frame_signal_matches_tf(self):
        x = np.random.default_rng(42).random((2, 97)).astype("float32")
        for frame_length, frame_step in ((10, 5), (16, 16), (32, 3)):
            self.assertAllClose(
                frame_signal(x, frame_length, frame_step),
                tf.signal.frame(x, frame_length, frame_step, pad_end=False),
            )

    def test_frame_signal_shorter_than_frame(self):
        x = np.ones((1, 3), dtype="float32")
        frames = frame_signal(x, 8, 2)
        self.assertEqual(frames.shape, (1, 0, 8))
