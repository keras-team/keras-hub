"""Import and initialize Keras-Hub export functionality.

This module automatically extends Keras-Hub models with export capabilities
when imported.
"""

# Import the exporters functionality
try:
    from keras_hub.src.exporters import *
    # The __init__.py file automatically adds the export method to Task base class
except ImportError as e:
    print(f"⚠️  Failed to import Keras-Hub export functionality: {e}")
