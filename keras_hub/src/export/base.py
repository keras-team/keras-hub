"""Base classes for Keras-Hub model exporters.

This module provides the foundation for exporting Keras-Hub models to various
formats. It defines the abstract base classes that all exporters must implement.
"""

from abc import ABC
from abc import abstractmethod


class KerasHubExporterConfig(ABC):
    """Base configuration class for Keras-Hub model exporters.

    This class defines the interface for exporter configurations that specify
    how different types of Keras-Hub models should be exported.
    """

    # Model type this exporter handles (e.g., "causal_lm", "text_classifier")
    MODEL_TYPE = None

    # Expected input structure for this model type
    EXPECTED_INPUTS = []

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
        if isinstance(param, dict):
            input_signature = param
        else:
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
