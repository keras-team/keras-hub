"""Registry initialization for Keras-Hub export functionality.

This module initializes the export registry with available configurations and
exporters.
"""

from keras_hub.src.export.base import ExporterRegistry
from keras_hub.src.export.configs import CausalLMExporterConfig
from keras_hub.src.export.configs import ImageClassifierExporterConfig
from keras_hub.src.export.configs import ImageSegmenterExporterConfig
from keras_hub.src.export.configs import ObjectDetectorExporterConfig
from keras_hub.src.export.configs import Seq2SeqLMExporterConfig
from keras_hub.src.export.configs import TextClassifierExporterConfig
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier


def initialize_export_registry():
    """Initialize the export registry with available configurations and
    exporters."""
    # Register configurations for different model types using classes
    # NOTE: Seq2SeqLM must be registered before CausalLM since it's a subclass
    ExporterRegistry.register_config(Seq2SeqLM, Seq2SeqLMExporterConfig)
    ExporterRegistry.register_config(CausalLM, CausalLMExporterConfig)
    ExporterRegistry.register_config(
        TextClassifier, TextClassifierExporterConfig
    )

    # Register vision model configurations
    ExporterRegistry.register_config(
        ImageClassifier, ImageClassifierExporterConfig
    )
    ExporterRegistry.register_config(
        ObjectDetector, ObjectDetectorExporterConfig
    )
    ExporterRegistry.register_config(
        ImageSegmenter, ImageSegmenterExporterConfig
    )

    # Register exporters for different formats
    try:
        from keras_hub.src.export.litert import LiteRTExporter

        ExporterRegistry.register_exporter("litert", LiteRTExporter)
    except ImportError:
        # Litert not available
        pass


def export_model(model, filepath, format="litert", **kwargs):
    """Export a Keras-Hub model to the specified format.

    This is the main export function that automatically detects the model type
    and uses the appropriate exporter configuration.

    Args:
        model: The Keras-Hub model to export
        filepath: Path where to save the exported model (without extension)
        format: Export format (currently supports "litert")
        **kwargs: Additional arguments passed to the exporter
    """
    # Registry is initialized at module level
    config = ExporterRegistry.get_config_for_model(model)

    # Get the exporter for the specified format
    exporter = ExporterRegistry.get_exporter(format, config, **kwargs)

    # Export the model
    exporter.export(filepath)


def extend_export_method_for_keras_hub():
    """Extend the export method for Keras-Hub models to handle dictionary
    inputs."""
    try:
        import keras

        from keras_hub.src.models.task import Task

        # Store the original export method if it exists
        original_export = getattr(Task, "export", None) or getattr(
            keras.Model, "export", None
        )

        def keras_hub_export(
            self,
            filepath,
            format="litert",
            verbose=False,
            **kwargs,
        ):
            """Extended export method for Keras-Hub models.

            This method extends Keras' export functionality to properly handle
            Keras-Hub models that expect dictionary inputs.

            Args:
                filepath: Path where to save the exported model (without
                    extension)
                format: Export format. Supports "litert", "tf_saved_model",
                    etc.
                verbose: Whether to print verbose output during export
                **kwargs: Additional arguments passed to the exporter
            """
            # Check if this is a Keras-Hub model that needs special handling
            if format == "litert" and self._is_keras_hub_model():
                # Use our Keras-Hub specific export logic
                kwargs["verbose"] = verbose
                export_model(self, filepath, format=format, **kwargs)
            else:
                # Fall back to the original Keras export method
                if original_export:
                    original_export(
                        self, filepath, format=format, verbose=verbose, **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"Export format '{format}' not supported for this "
                        "model type"
                    )

        def _is_keras_hub_model(self):
            """Check if this model is a Keras-Hub model that needs special
            handling.

            Since this method is monkey-patched onto the Task class, `self`
            will always be an instance of a Task subclass from keras_hub.
            """
            return isinstance(self, Task)

        # Add the helper method and export method to the Task class
        Task._is_keras_hub_model = _is_keras_hub_model
        Task.export = keras_hub_export

    except ImportError:
        # Task class not available, skip extension
        pass
    except Exception as e:
        # Log error but don't fail import
        import warnings

        warnings.warn(
            f"Failed to extend export method for Keras-Hub models: {e}"
        )


# Initialize the registry when this module is imported
initialize_export_registry()
extend_export_method_for_keras_hub()
