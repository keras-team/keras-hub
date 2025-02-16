import keras

from keras_hub.src.models.moonshine.moonshine_preprocessor import (
    MoonshinePreprocessor,
)


# Skipped for now (not subclassed from unittest.TestCase).
class MoonshinePreprocessorTest:
    def setUp(self):
        self.dim = 256
        self.preprocessor = MoonshinePreprocessor(dim=self.dim)
        self.dummy_audio = keras.ops.convert_to_tensor(
            [[0.1] * 16000], dtype="float32"
        )
        self.dummy_audio = keras.ops.expand_dims(self.dummy_audio, axis=-1)

    def test_output_shape(self):
        features = self.preprocessor(self.dummy_audio)
        features_shape = keras.ops.shape(features)
        features_shape_np = keras.ops.convert_to_numpy(features_shape)
        self.assertEqual(tuple(features_shape_np), (1, 40, self.dim))

    def test_get_config(self):
        config = self.preprocessor.get_config()
        self.assertIsInstance(config, dict)
        if "dim" in config:
            self.assertEqual(config["dim"], self.dim)
