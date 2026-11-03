from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.serialization import deserialize
from keras_hub.src.kv_press.serialization import get
from keras_hub.src.kv_press.serialization import serialize
from keras_hub.src.tests.test_case import TestCase


class SerializationTest(TestCase):
    def test_serialization(self):
        press = KnormPress(compression_ratio=0.7)
        restored = deserialize(serialize(press))
        self.assertDictEqual(press.get_config(), restored.get_config())

    def test_get(self):
        # Test get from string.
        identifier = "knorm"
        press = get(identifier)
        self.assertIsInstance(press, KnormPress)

        # Test dict identifier.
        original_press = KnormPress(compression_ratio=0.7)
        config = serialize(original_press)
        restored_press = get(config)
        self.assertDictEqual(
            serialize(restored_press),
            serialize(original_press),
        )

        # Test identifier is already a press instance.
        original_press = KnormPress(compression_ratio=0.7)
        restored_press = get(original_press)
        self.assertEqual(original_press, restored_press)

        # Test identifier is None.
        self.assertIsNone(get(None))
