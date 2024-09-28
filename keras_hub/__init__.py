import os

# sentencepiece segfaults on some version of tensorflow if tf is imported first.
try:
    import sentencepiece
except ImportError:
    pass

# Import everything from /api/ into keras.
from keras_hub.api import *  # noqa: F403
from keras_hub.api import __version__  # Import * ignores names start with "_".

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Don't pollute namespace.
del os


# Never autocomplete `.src` or `.api` on an imported keras object.
def __dir__():
    keys = dict.fromkeys((globals().keys()))
    keys.pop("src")
    keys.pop("api")
    return list(keys)


# Don't import `.src` or `.api` during `from keras import *`.
__all__ = [
    name
    for name in globals().keys()
    if not (name.startswith("_") or name in ("src", "api"))
]
