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


def add_export_method_to_task():
    """Add the export method to the Task base class."""
    try:
        from keras_hub.src.models.task import Task
        
        def export(self, filepath: str, format: str = "lite_rt", **kwargs) -> None:
            """Export the model to the specified format.
            
            Args:
                filepath: str. Path where to save the exported model (without extension)
                format: str. Export format. Currently supports "lite_rt"
                **kwargs: Additional arguments passed to the exporter
            """
            export_model(self, filepath, format=format, **kwargs)
        
        # Add the method to the Task class if it doesn't exist
        if not hasattr(Task, 'export'):
            Task.export = export
            
    except Exception as e:
        print(f"⚠️  Failed to add export method to Task class: {e}")


# Initialize the registry when this module is imported
initialize_export_registry()
add_export_method_to_task()
