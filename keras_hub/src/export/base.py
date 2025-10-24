"""Base classes for Keras-Hub model exporters.

This module provides the foundation for exporting Keras-Hub models to various
formats. It follows the Optimum pattern of having different exporters for
different model types and formats.
"""

from abc import ABC
from abc import abstractmethod

# Import model classes for registry
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier


class KerasHubExporterConfig(ABC):
    """Base configuration class for Keras-Hub model exporters.

    This class defines the interface for exporter configurations that specify
    how different types of Keras-Hub models should be exported.
    """

    # Model type this exporter handles (e.g., "causal_lm", "text_classifier")
    MODEL_TYPE = None

    # Expected input structure for this model type
    EXPECTED_INPUTS = []

    # Default sequence length if not specified
    DEFAULT_SEQUENCE_LENGTH = 128

    def __init__(self, model, **kwargs):
        """Initialize the exporter configuration.

        Args:
            model: `keras.Model`. The Keras-Hub model to export.
            **kwargs: Additional configuration parameters.
        """
        self.model = model
        self.config_kwargs = kwargs
        self._validate_model()

    def _validate_model(self):
        """Validate that the model is compatible with this exporter."""
        if not self._is_model_compatible():
            raise ValueError(
                f"Model {self.model.__class__.__name__} is not compatible "
                f"with {self.__class__.__name__}"
            )

    @abstractmethod
    def _is_model_compatible(self):
        """Check if the model is compatible with this exporter.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        pass

    @abstractmethod
    def get_input_signature(self, sequence_length=None):
        """Get the input signature for this model type.

        Args:
            sequence_length: `int` or `None`. Optional sequence length for
                input tensors.

        Returns:
            `dict`. Dictionary mapping input names to tensor specifications.
        """
        pass


class KerasHubExporter(ABC):
    """Base class for Keras-Hub model exporters.

    This class provides the common interface for exporting Keras-Hub models
    to different formats (LiteRT, ONNX, etc.).
    """

    def __init__(self, config, **kwargs):
        """Initialize the exporter.

        Args:
            config: `KerasHubExporterConfig`. Exporter configuration specifying
                model type and parameters.
            **kwargs: Additional exporter-specific parameters.
        """
        self.config = config
        self.model = config.model
        self.export_kwargs = kwargs

    @abstractmethod
    def export(self, filepath):
        """Export the model to the specified filepath.

        Args:
            filepath: `str`. Path where to save the exported model.
        """
        pass

    def _ensure_model_built(self, param=None):
        """Ensure the model is properly built with correct input structure.

        This method builds the model using model.build() with input shapes.
        This creates the necessary variables and initializes the model structure
        for export without needing dummy data.

        Args:
            param: `int` or `None`. Optional parameter for input signature
                (e.g., sequence_length for text models, image_size for vision
                models).
        """
        # Get input signature (returns dict of InputSpec objects)
        input_signature = self.config.get_input_signature(param)

        # Extract shapes from InputSpec objects
        input_shapes = {}
        for name, spec in input_signature.items():
            if hasattr(spec, "shape"):
                input_shapes[name] = spec.shape
            else:
                # Fallback for unexpected formats
                input_shapes[name] = spec

        # Build the model using shapes only (no actual data allocation)
        # This creates variables and initializes the model structure
        self.model.build(input_shape=input_shapes)


class ExporterRegistry:
    """Registry for mapping model types to their appropriate exporters."""

    _configs = {}
    _exporters = {}

    @classmethod
    def register_config(cls, model_class, config_class):
        """Register a configuration class for a model type.

        Args:
            model_class: `type`. The model class (e.g., CausalLM)
            config_class: `type`. The configuration class
        """
        cls._configs[model_class] = config_class

    @classmethod
    def register_exporter(cls, format_name, exporter_class):
        """Register an exporter class for a format.

        Args:
            format_name: `str`. The export format (e.g., "litert")
            exporter_class: `type`. The exporter class
        """
        cls._exporters[format_name] = exporter_class

    @classmethod
    def get_config_for_model(cls, model):
        """Get the appropriate configuration for a model.

        Args:
            model: `keras.Model`. The Keras-Hub model

        Returns:
            `KerasHubExporterConfig`. An appropriate exporter configuration
            instance

        Raises:
            ValueError: If no configuration is found for the model type
        """
        # Find the matching model class
        for model_class in [
            CausalLM,
            TextClassifier,
            Seq2SeqLM,
            ImageClassifier,
            ObjectDetector,
            ImageSegmenter,
        ]:
            if isinstance(model, model_class):
                if model_class not in cls._configs:
                    raise ValueError(
                        f"No configuration found for model type: "
                        f"{model_class.__name__}"
                    )
                config_class = cls._configs[model_class]
                return config_class(model)

        # If we get here, model type is not recognized
        raise ValueError(
            f"Could not detect model type for {model.__class__.__name__}. "
            "Supported types: CausalLM, TextClassifier, Seq2SeqLM, "
            "ImageClassifier, ObjectDetector, ImageSegmenter"
        )

    @classmethod
    def get_exporter(cls, format_name, config, **kwargs):
        """Get an exporter for the specified format.

        Args:
            format_name: `str`. The export format
            config: `KerasHubExporterConfig`. The exporter configuration
            **kwargs: `dict`. Additional parameters for the exporter

        Returns:
            `KerasHubExporter`. An appropriate exporter instance

        Raises:
            ValueError: If no exporter is found for the format
        """
        if format_name not in cls._exporters:
            raise ValueError(f"No exporter found for format: {format_name}")

        exporter_class = cls._exporters[format_name]
        return exporter_class(config, **kwargs)
