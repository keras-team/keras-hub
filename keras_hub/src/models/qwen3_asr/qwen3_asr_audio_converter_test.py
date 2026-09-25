import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioConverterTest(TestCase):
    def test_serialization(self):
        self.run_layer_test(
            cls=Qwen3ASRAudioConverter,
            init_kwargs={
                "num_mels": 4,
                "max_audio_length": 1,
                "min_length": 8000,
            },
            input_data=np.ones((1, 8000), dtype="float32"),
            expected_output_shape=(1, 50, 4),
            run_training_check=False,
            run_precision_checks=False,
        )

    def test_converter_call(self):
        # 1 second of audio at 16000Hz
        audio = np.random.uniform(size=(16000,))

        converter = Qwen3ASRAudioConverter(
            num_mels=128,
            sampling_rate=16000,
            max_audio_length=30,
            n_window=50,
        )

        # Call with unbatched input
        output = converter(audio)
        # The output is padded to a multiple of 100 mel frames.
        self.assertEqual(output.shape, (100, 128))

        # Call with batched input
        batched_audio = np.random.uniform(size=(2, 16000))
        output_batched = converter(batched_audio)
        self.assertEqual(output_batched.shape, (2, 100, 128))

    def test_audio_shape(self):
        converter = Qwen3ASRAudioConverter(
            max_audio_length=30,
            n_window=50,
        )
        self.assertEqual(converter.audio_shape(), (None, 128))

        # Test with custom max_audio_length not a multiple of 100 frames.
        # e.g., max_audio_length = 1.05s -> 16800 samples.
        # stride = 160 -> 16800 // 160 = 105 frames.
        # Padded to multiple of 100 -> 200 frames.
        converter_short = Qwen3ASRAudioConverter(
            max_audio_length=1.05,
            n_window=50,
        )
        self.assertEqual(converter_short.audio_shape(), (None, 128))

    def test_variable_length_batched_input(self):
        converter = Qwen3ASRAudioConverter(
            max_audio_length=1.05,
            n_window=50,
        )
        audio = [np.ones((16000,)), np.ones((8000,))]

        output = converter(audio)

        self.assertEqual(output.shape, (2, 100, 128))

    def test_min_length_padding(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)

        output = converter(np.ones((4000,)))

        self.assertEqual(output.shape, (50, 128))

    def test_python_path(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)
        audio = [np.ones((4000,)), np.ones((8000,))]

        output = converter._call_python(audio)

        self.assertEqual(output.shape, (2, 50, 128))

    def test_audio_longer_than_default_max_length(self):
        converter = Qwen3ASRAudioConverter(n_window=50)
        audio = [np.zeros((5 * 16000,)), np.zeros((31 * 16000,))]

        output = converter(audio)

        self.assertEqual(output.shape, (2, 3100, 128))
