"""Tests for base export classes."""

import keras

from keras_hub.src.export.base import KerasHubExporter
from keras_hub.src.export.base import KerasHubExporterConfig
from keras_hub.src.tests.test_case import TestCase


class DummyExporterConfig(KerasHubExporterConfig):
    """Dummy configuration for testing."""

    MODEL_TYPE = "test_model"
    EXPECTED_INPUTS = ["input_ids", "attention_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def __init__(self, model, compatible=True, **kwargs):
        self.is_compatible = compatible
        super().__init__(model, **kwargs)

    def _is_model_compatible(self):
        return self.is_compatible

    def get_input_signature(self, sequence_length=None):
        seq_len = sequence_length or self.DEFAULT_SEQUENCE_LENGTH
        return {
            "input_ids": keras.layers.InputSpec(
                shape=(None, seq_len), dtype="int32"
            ),
            "attention_mask": keras.layers.InputSpec(
                shape=(None, seq_len), dtype="int32"
            ),
        }


class DummyExporter(KerasHubExporter):
    """Dummy exporter for testing."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.exported = False
        self.export_path = None

    def export(self, filepath):
        self.exported = True
        self.export_path = filepath
        return filepath


class KerasHubExporterConfigTest(TestCase):
    """Tests for KerasHubExporterConfig base class."""

    def test_init_with_compatible_model(self):
        """Test initialization with a compatible model."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model, compatible=True)

        self.assertEqual(config.model, model)
        self.assertEqual(config.MODEL_TYPE, "test_model")
        self.assertEqual(
            config.EXPECTED_INPUTS, ["input_ids", "attention_mask"]
        )

    def test_init_with_incompatible_model_raises_error(self):
        """Test that incompatible model raises ValueError."""
        model = keras.Sequential([keras.layers.Dense(10)])

        with self.assertRaisesRegex(ValueError, "not compatible"):
            DummyExporterConfig(model, compatible=False)

    def test_get_input_signature_default_sequence_length(self):
        """Test get_input_signature with default sequence length."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)

        signature = config.get_input_signature()

        self.assertIn("input_ids", signature)
        self.assertIn("attention_mask", signature)
        self.assertEqual(signature["input_ids"].shape, (None, 128))
        self.assertEqual(signature["attention_mask"].shape, (None, 128))

    def test_get_input_signature_custom_sequence_length(self):
        """Test get_input_signature with custom sequence length."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)

        signature = config.get_input_signature(sequence_length=256)

        self.assertEqual(signature["input_ids"].shape, (None, 256))
        self.assertEqual(signature["attention_mask"].shape, (None, 256))

    def test_config_kwargs_stored(self):
        """Test that additional kwargs are stored."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(
            model, custom_param="value", another_param=42
        )

        self.assertEqual(config.config_kwargs["custom_param"], "value")
        self.assertEqual(config.config_kwargs["another_param"], 42)


class KerasHubExporterTest(TestCase):
    """Tests for KerasHubExporter base class."""

    def test_init_stores_config_and_model(self):
        """Test that initialization stores config and model."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)
        exporter = DummyExporter(config, verbose=True, custom_param="test")

        self.assertEqual(exporter.config, config)
        self.assertEqual(exporter.model, model)
        self.assertEqual(exporter.export_kwargs["verbose"], True)
        self.assertEqual(exporter.export_kwargs["custom_param"], "test")

    def test_export_method_called(self):
        """Test that export method can be called."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)
        exporter = DummyExporter(config)

        result = exporter.export("/tmp/test_model")

        self.assertTrue(exporter.exported)
        self.assertEqual(exporter.export_path, "/tmp/test_model")
        self.assertEqual(result, "/tmp/test_model")
