"""Registry initialization for Keras-Hub export functionality.

This module initializes the export registry with available configurations and
exporters.
"""

from keras_hub.src.export.base import ExporterRegistry
from keras_hub.src.export.configs import CausalLMExporterConfig
from keras_hub.src.export.configs import Seq2SeqLMExporterConfig
from keras_hub.src.export.configs import TextClassifierExporterConfig
from keras_hub.src.export.configs import TextModelExporterConfig


def initialize_export_registry():
    """Initialize the export registry with available configurations and
    exporters."""
    # Register configurations for different model types
    ExporterRegistry.register_config("causal_lm", CausalLMExporterConfig)
    ExporterRegistry.register_config(
        "text_classifier", TextClassifierExporterConfig
    )
    ExporterRegistry.register_config("seq2seq_lm", Seq2SeqLMExporterConfig)
    ExporterRegistry.register_config("text_model", TextModelExporterConfig)

    # Register exporters for different formats
    try:
        from keras_hub.src.export.lite_rt import LiteRTExporter

        ExporterRegistry.register_exporter("lite_rt", LiteRTExporter)
    except ImportError:
        # LiteRT not available
        pass


def export_model(model, filepath, format="lite_rt", **kwargs):
    """Export a Keras-Hub model to the specified format.

    This is the main export function that automatically detects the model type
    and uses the appropriate exporter configuration.

    Args:
        model: The Keras-Hub model to export
        filepath: Path where to save the exported model (without extension)
        format: Export format (currently supports "lite_rt")
        **kwargs: Additional arguments passed to the exporter
    """
    # Ensure registry is initialized
    initialize_export_registry()

    # Get the appropriate configuration for this model
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
            format="lite_rt",
            verbose=False,
            **kwargs,
        ):
            """Extended export method for Keras-Hub models.

            This method extends Keras' export functionality to properly handle
            Keras-Hub models that expect dictionary inputs.

            Args:
                filepath: Path where to save the exported model (without
                    extension)
                format: Export format. Supports "lite_rt", "tf_saved_model",
                    etc.
                verbose: Whether to print verbose output during export
                **kwargs: Additional arguments passed to the exporter
            """
            # Check if this is a Keras-Hub model that needs special handling
            if format == "lite_rt" and self._is_keras_hub_model():
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
            handling."""
            if hasattr(self, "__class__"):
                class_name = self.__class__.__name__
                module_name = self.__class__.__module__

                # Check if it's from keras_hub package
                if "keras_hub" in module_name:
                    return True

                # Check if it has keras-hub specific attributes
                if hasattr(self, "preprocessor") and hasattr(self, "backbone"):
                    return True

                # Check for common Keras-Hub model names
                keras_hub_model_names = [
                    "CausalLM",
                    "Seq2SeqLM",
                    "TextClassifier",
                    "ImageClassifier",
                ]
                if any(name in class_name for name in keras_hub_model_names):
                    return True

            return False

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
