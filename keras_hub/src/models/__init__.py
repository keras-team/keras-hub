"""Import and initialize Keras-Hub export functionality.

This module automatically extends Keras-Hub models with export capabilities
when imported.
"""

# Import the export functionality
try:
    from keras_hub.src.export.registry import add_export_method_to_task
    from keras_hub.src.export.registry import initialize_export_registry
    # Initialize export functionality
    initialize_export_registry()
    add_export_method_to_task()
except ImportError as e:
    print(f"⚠️  Failed to import Keras-Hub export functionality: {e}")
