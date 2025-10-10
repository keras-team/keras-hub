"""Base classes for Keras-Hub model exporters.

This module provides the foundation for exporting Keras-Hub models to various
formats. It follows the Optimum pattern of having different exporters for
different model types and formats.
"""

from abc import ABC
from abc import abstractmethod

try:
    import keras

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = None


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
            bool: True if compatible, False otherwise
        """
        pass

    @abstractmethod
    def get_input_signature(self, sequence_length=None):
        """Get the input signature for this model type.

        Args:
            sequence_length: `int` or `None`. Optional sequence length for
                input tensors.

        Returns:
            A dictionary mapping input names to their tensor specifications.
        """
        pass

    def get_dummy_inputs(self, sequence_length=None):
        """Generate dummy inputs for model building and testing.

        Args:
            sequence_length: `int` or `None`. Optional sequence length for
                dummy inputs.

        Returns:
            A dictionary of dummy inputs.
        """
        if sequence_length is None:
            sequence_length = self.DEFAULT_SEQUENCE_LENGTH

        dummy_inputs = {}

        # Common inputs for most Keras-Hub models
        if "token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["token_ids"] = keras.ops.ones(
                (1, sequence_length), dtype="int32"
            )
        if "padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["padding_mask"] = keras.ops.ones(
                (1, sequence_length), dtype="bool"
            )

        # Encoder-decoder specific inputs
        if "encoder_token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["encoder_token_ids"] = keras.ops.ones(
                (1, sequence_length), dtype="int32"
            )
        if "encoder_padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["encoder_padding_mask"] = keras.ops.ones(
                (1, sequence_length), dtype="bool"
            )
        if "decoder_token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["decoder_token_ids"] = keras.ops.ones(
                (1, sequence_length), dtype="int32"
            )
        if "decoder_padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["decoder_padding_mask"] = keras.ops.ones(
                (1, sequence_length), dtype="bool"
            )

        return dummy_inputs


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
        for export, avoiding the need for dummy forward passes.

        Note: We don't check model.built because it can be True even if the
        model isn't properly initialized with the correct input structure.

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

        try:
            # Build the model using shapes only (no actual data allocation)
            # This creates variables and initializes the model structure
            self.model.build(input_shape=input_shapes)
        except Exception as e:
            # Fallback to forward pass approach if build() fails
            # This maintains backward compatibility for models that don't
            # support shape-based building
            try:
                dummy_inputs = self.config.get_dummy_inputs(param)
                _ = self.model(dummy_inputs, training=False)
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to build model with both shape-based building "
                    f"({e}) and forward pass ({fallback_error}). Please ensure "
                    f"the model is properly constructed."
                )


class ExporterRegistry:
    """Registry for mapping model types to their appropriate exporters."""

    _configs = {}
    _exporters = {}

    @classmethod
    def register_config(cls, model_type, config_class):
        """Register a configuration class for a model type.

        Args:
            model_type: The model type (e.g., "causal_lm")
            config_class: The configuration class
        """
        cls._configs[model_type] = config_class

    @classmethod
    def register_exporter(cls, format_name, exporter_class):
        """Register an exporter class for a format.

        Args:
            format_name: The export format (e.g., "litert")
            exporter_class: The exporter class
        """
        cls._exporters[format_name] = exporter_class

    @classmethod
    def get_config_for_model(cls, model):
        """Get the appropriate configuration for a model.

        Args:
            model: The Keras-Hub model

        Returns:
            KerasHubExporterConfig: An appropriate exporter configuration
            instance

        Raises:
            ValueError: If no configuration is found for the model type
        """
        model_type = cls._detect_model_type(model)

        if model_type not in cls._configs:
            raise ValueError(
                f"No configuration found for model type: {model_type}"
            )

        config_class = cls._configs[model_type]
        return config_class(model)

    @classmethod
    def get_exporter(cls, format_name, config, **kwargs):
        """Get an exporter for the specified format.

        Args:
            format_name: The export format
            config: The exporter configuration
            **kwargs: Additional parameters for the exporter

        Returns:
            KerasHubExporter: An appropriate exporter instance

        Raises:
            ValueError: If no exporter is found for the format
        """
        if format_name not in cls._exporters:
            raise ValueError(f"No exporter found for format: {format_name}")

        exporter_class = cls._exporters[format_name]
        return exporter_class(config, **kwargs)

    @classmethod
    def _detect_model_type(cls, model):
        """Detect the model type from the model instance.

        Args:
            model: The Keras-Hub model

        Returns:
            str: The detected model type
        """
        # Import here to avoid circular imports
        try:
            from keras_hub.src.models.causal_lm import CausalLM
            from keras_hub.src.models.image_segmenter import ImageSegmenter
            from keras_hub.src.models.object_detector import ObjectDetector
            from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
        except ImportError:
            CausalLM = None
            Seq2SeqLM = None
            ObjectDetector = None
            ImageSegmenter = None

        model_class_name = model.__class__.__name__

        if CausalLM and isinstance(model, CausalLM):
            return "causal_lm"
        elif "TextClassifier" in model_class_name:
            return "text_classifier"
        elif Seq2SeqLM and isinstance(model, Seq2SeqLM):
            return "seq2seq_lm"
        elif "ImageClassifier" in model_class_name:
            return "image_classifier"
        elif ObjectDetector and isinstance(model, ObjectDetector):
            return "object_detector"
        elif "ObjectDetector" in model_class_name:
            return "object_detector"
        elif ImageSegmenter and isinstance(model, ImageSegmenter):
            return "image_segmenter"
        elif "ImageSegmenter" in model_class_name:
            return "image_segmenter"
        else:
            # Default to text model for generic Keras-Hub models
            return "text_model"
