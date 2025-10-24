"""Tests for export registry functionality."""

import keras

from keras_hub.src.export.base import ExporterRegistry
from keras_hub.src.export.base import KerasHubExporter
from keras_hub.src.export.base import KerasHubExporterConfig
from keras_hub.src.export.configs import CausalLMExporterConfig
from keras_hub.src.export.configs import ImageClassifierExporterConfig
from keras_hub.src.export.configs import TextClassifierExporterConfig
from keras_hub.src.export.registry import initialize_export_registry
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class DummyExporterConfig(KerasHubExporterConfig):
    """Dummy config for testing."""

    MODEL_TYPE = "test_model"
    EXPECTED_INPUTS = ["input_1"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        return True

    def get_input_signature(self, sequence_length=None):
        seq_len = sequence_length or self.DEFAULT_SEQUENCE_LENGTH
        return {
            "input_1": keras.layers.InputSpec(
                shape=(None, seq_len), dtype="int32"
            )
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


class ExporterRegistryTest(TestCase):
    """Tests for ExporterRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Clear registry before each test
        ExporterRegistry._configs = {}
        ExporterRegistry._exporters = {}

    def test_register_and_retrieve_config(self):
        """Test registering and retrieving a configuration."""

        # Create a dummy model class
        class DummyModel(keras.Model):
            pass

        # Register configuration
        ExporterRegistry.register_config(DummyModel, DummyExporterConfig)

        # Verify registration
        self.assertIn(DummyModel, ExporterRegistry._configs)
        self.assertEqual(
            ExporterRegistry._configs[DummyModel], DummyExporterConfig
        )

    def test_register_and_retrieve_exporter(self):
        """Test registering and retrieving an exporter."""
        # Register exporter
        ExporterRegistry.register_exporter("test_format", DummyExporter)

        # Verify registration
        self.assertIn("test_format", ExporterRegistry._exporters)
        self.assertEqual(
            ExporterRegistry._exporters["test_format"], DummyExporter
        )

    def test_get_exporter_creates_instance(self):
        """Test that get_exporter creates an exporter instance."""
        # Register exporter
        ExporterRegistry.register_exporter("test_format", DummyExporter)

        # Create a dummy config
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)

        # Get exporter
        exporter = ExporterRegistry.get_exporter(
            "test_format", config, test_param="value"
        )

        # Verify it's an instance of the correct class
        self.assertIsInstance(exporter, DummyExporter)
        self.assertEqual(exporter.config, config)
        self.assertEqual(exporter.export_kwargs["test_param"], "value")

    def test_get_exporter_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        model = keras.Sequential([keras.layers.Dense(10)])
        config = DummyExporterConfig(model)

        with self.assertRaisesRegex(ValueError, "No exporter found for format"):
            ExporterRegistry.get_exporter("invalid_format", config)

    def test_get_config_for_model_with_unknown_type_raises_error(self):
        """Test that unknown model type raises ValueError."""
        # Initialize registry with known types
        initialize_export_registry()

        # Create a generic Keras model (not a Keras-Hub model)
        model = keras.Sequential([keras.layers.Dense(10)])

        with self.assertRaisesRegex(ValueError, "Could not detect model type"):
            ExporterRegistry.get_config_for_model(model)

    def test_initialize_export_registry(self):
        """Test that initialize_export_registry registers all configs."""
        initialize_export_registry()

        # Check that model configurations are registered
        self.assertIn(CausalLM, ExporterRegistry._configs)
        self.assertIn(TextClassifier, ExporterRegistry._configs)
        self.assertIn(ImageClassifier, ExporterRegistry._configs)

        # Check that the correct config classes are registered
        self.assertEqual(
            ExporterRegistry._configs[CausalLM], CausalLMExporterConfig
        )
        self.assertEqual(
            ExporterRegistry._configs[TextClassifier],
            TextClassifierExporterConfig,
        )
        self.assertEqual(
            ExporterRegistry._configs[ImageClassifier],
            ImageClassifierExporterConfig,
        )

        # Check that litert exporter is registered (if available)
        if "litert" in ExporterRegistry._exporters:
            self.assertIn("litert", ExporterRegistry._exporters)


class ExportModelFunctionTest(TestCase):
    """Tests for export_model convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Clear and reinitialize registry
        ExporterRegistry._configs = {}
        ExporterRegistry._exporters = {}
        ExporterRegistry.register_exporter("test_format", DummyExporter)

    def test_get_config_requires_known_model_type(self):
        """Test that get_config_for_model only works with known types.

        Note: This test documents current behavior. The registry could be
        improved to support dynamically registered model types.
        See code review item #3 about redundant model type detection.
        """

        # Create a generic Keras model
        class GenericModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(10)

            def call(self, inputs):
                return self.dense(inputs)

        model = GenericModel()
        model.build(input_shape=(None, 128))

        # This should raise ValueError for unknown model type
        with self.assertRaisesRegex(ValueError, "Could not detect model type"):
            ExporterRegistry.get_config_for_model(model)
