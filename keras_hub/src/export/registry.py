"""Registry initialization for Keras-Hub export functionality.

This module initializes the export registry with available configurations and exporters.
"""

from keras_hub.src.export.base import ExporterRegistry
from keras_hub.src.export.configs import (
    CausalLMExporterConfig,
    TextClassifierExporterConfig,
    Seq2SeqLMExporterConfig,
    TextModelExporterConfig
)
from keras_hub.src.export.lite_rt import LiteRTExporter


def initialize_export_registry():
    """Initialize the export registry with available configurations and exporters."""
    # Register configurations for different model types
    ExporterRegistry.register_config("causal_lm", CausalLMExporterConfig)
    ExporterRegistry.register_config("text_classifier", TextClassifierExporterConfig)
    ExporterRegistry.register_config("seq2seq_lm", Seq2SeqLMExporterConfig)
    ExporterRegistry.register_config("text_model", TextModelExporterConfig)

    # Register exporters for different formats
    ExporterRegistry.register_exporter("lite_rt", LiteRTExporter)


def export_model(model, filepath: str, format: str = "lite_rt", **kwargs):
    """Export a Keras-Hub model to the specified format.
    
    This is the main export function that automatically detects the model type
    and uses the appropriate exporter configuration.
    
    Args:
        model: The Keras-Hub model to export
        filepath: Path where to save the exported model (without extension)
        format: Export format (currently supports "lite_rt")
        **kwargs: Additional arguments passed to the exporter (including verbose)
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
    """Extend the export method for Keras-Hub models to handle dictionary inputs."""
    try:
        from keras_hub.src.models.task import Task
        import keras
        
        # Store the original export method
        original_export = Task.export if hasattr(Task, 'export') else keras.Model.export
        
        def keras_hub_export(self, filepath: str, format: str = "lite_rt", verbose=None, **kwargs):
            """Extended export method for Keras-Hub models.
            
            This method extends Keras' export functionality to properly handle
            Keras-Hub models that expect dictionary inputs.
            
            Args:
                filepath: str. Path where to save the exported model (without extension)
                format: str. Export format. Supports "lite_rt", "tf_saved_model", etc.
                verbose: bool. Whether to print verbose output during export
                **kwargs: Additional arguments passed to the exporter
            """
            # Check if this is a Keras-Hub model that needs special handling
            if format == "lite_rt" and self._is_keras_hub_model():
                # Use our Keras-Hub specific export logic
                # Make sure we don't duplicate the verbose parameter
                if verbose is not None and 'verbose' not in kwargs:
                    kwargs['verbose'] = verbose
                export_model(self, filepath, format=format, **kwargs)
            else:
                # Fall back to the original Keras export method
                original_export(self, filepath, format=format, verbose=verbose, **kwargs)
        
        def _is_keras_hub_model(self):
            """Check if this model is a Keras-Hub model that needs special handling."""
            # Check if it's a Task (most Keras-Hub models inherit from Task)
            if hasattr(self, '__class__'):
                class_name = self.__class__.__name__
                module_name = self.__class__.__module__
                
                # Check if it's from keras_hub package
                if 'keras_hub' in module_name:
                    return True
                    
                # Check if it has keras-hub specific attributes
                if hasattr(self, 'preprocessor') and hasattr(self, 'backbone'):
                    return True
                    
                # Check for common Keras-Hub model names
                keras_hub_model_names = ['CausalLM', 'Seq2SeqLM', 'TextClassifier', 'ImageClassifier']
                if any(name in class_name for name in keras_hub_model_names):
                    return True
                    
            return False
        
        # Add the helper method to the class
        Task._is_keras_hub_model = _is_keras_hub_model
        
        # Override the export method
        Task.export = keras_hub_export
        
        print("✅ Extended export method for Keras-Hub models")
        
    except Exception as e:
        print(f"⚠️  Failed to extend export method for Keras-Hub models: {e}")
        import traceback
        traceback.print_exc()


# Initialize the registry when this module is imported
initialize_export_registry()
extend_export_method_for_keras_hub()
