import os

# Add everything in /api/ to the module search path.
import keras_hub

# Add everything in /api/ to the module search path.
__path__.extend(keras_hub.__path__)  # noqa: F405

from keras_hub import *  # noqa: F403, E402
from keras_hub import __version__ as __version__  # noqa: E402

# Don't pollute namespace.
del os
