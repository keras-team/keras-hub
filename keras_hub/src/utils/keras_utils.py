import sys

import keras
from absl import logging

try:
    import tensorflow as tf
except ImportError:
    tf = None


def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    # Copied from core Keras.
    if keras.utils.is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)


# Register twice for backwards compat.
@keras.saving.register_keras_serializable(package="keras_hub")
@keras.saving.register_keras_serializable(package="keras_nlp")
def gelu_approximate(x):
    return keras.activations.gelu(x, approximate=True)


def standardize_data_format(data_format):
    if data_format is None:
        return keras.config.image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def has_flash_attention_support():
    if (
        hasattr(keras.config, "is_flash_attention_enabled")
        and keras.config.backend() == "jax"
    ):
        try:
            from jax.nn import dot_product_attention as dot_product_attention
        except ImportError:
            logging.warning(
                "Flash attention is not supported in your current JAX version. "
                "Please update it by following the official guide: "
                "https://jax.readthedocs.io/en/latest/installation.html"
            )
            return False
        return True
    else:
        return False
