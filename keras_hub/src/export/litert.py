"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for
Keras-Hub models, handling their unique input structures and requirements.

The exporter supports dynamic shape inputs by default, leveraging TFLite's
native capability to resize input tensors at runtime. When applicable parameters
are not specified, models are exported with flexible dimensions that can be
resized via `interpreter.resize_tensor_input()` before inference.
"""


# Convenience function for direct export
def export_litert(model, filepath, **kwargs):
    """Export a Keras-Hub model to Litert format.

    This is a convenience function that automatically detects the model type
    and exports it using the appropriate configuration.

    Args:
        model: `keras.Model`. The Keras-Hub model to export.
        filepath: `str`. Path where to save the model (without extension).
        **kwargs: `dict`. Additional arguments passed to exporter.
    """
    from keras.src.export.litert import export_litert as keras_export_litert

    from keras_hub.src.export.configs import get_exporter_config

    # Get the appropriate configuration for this model type
    config = get_exporter_config(model)

    # Get domain-specific input signature from config
    input_signature = config.get_input_signature()

    # Call Keras Core's export_litert directly
    keras_export_litert(
        model,
        filepath,
        input_signature=input_signature,
        **kwargs,
    )
