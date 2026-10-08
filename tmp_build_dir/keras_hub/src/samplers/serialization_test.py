from keras_hub.src.samplers.serialization import deserialize
from keras_hub.src.samplers.serialization import get
from keras_hub.src.samplers.serialization import serialize
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.tests.test_case import TestCase


class SerializationTest(TestCase):
    def test_serialization(self):
        sampler = TopKSampler(k=5)
        restored = deserialize(serialize(sampler))
        self.assertDictEqual(sampler.get_config(), restored.get_config())

    def test_get(self):
        # Test get from string.
        identifier = "top_k"
        sampler = get(identifier)
        self.assertIsInstance(sampler, TopKSampler)

        # Test dict identifier.
        original_sampler = TopKSampler(k=7)
        config = serialize(original_sampler)
        restored_sampler = get(config)
        self.assertDictEqual(
            serialize(restored_sampler),
            serialize(original_sampler),
        )

        # Test identifier is already a sampler instance.
        original_sampler = TopKSampler(k=7)
        restored_sampler = get(original_sampler)
        self.assertEqual(original_sampler, restored_sampler)
