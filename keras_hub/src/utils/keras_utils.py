import inspect
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


def fused_attention_op_available():
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
    elif (
        hasattr(keras.config, "is_flash_attention_enabled")
        and keras.config.backend() == "torch"
    ):
        try:
            from torch.backends.cuda import SDPAParams as SDPAParams
            from torch.backends.cuda import (
                can_use_flash_attention as can_use_flash_attention,
            )
        except ImportError:
            logging.warning(
                "Flash attention is not supported in your current PyTorch "
                "version. Please update it by following the official guide: "
                "https://pytorch.org/get-started/locally/"
            )
            return False
        return True
    else:
        return False


def running_on_tpu():
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        devices = jax.devices()
        return any(d.platform == "tpu" for d in devices)
    elif backend == "tensorflow":
        import tensorflow as tf

        return bool(tf.config.list_logical_devices("TPU"))
    elif backend == "torch":
        return False


def running_on_gpu():
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        devices = jax.devices()
        return any(d.platform == "gpu" for d in devices)
    elif backend == "tensorflow":
        import tensorflow as tf

        return bool(tf.config.list_logical_devices("GPU"))
    elif backend == "torch":
        import torch

        return torch.cuda.is_available()


def gpu_supports_fused_attention_op():
    deny_list = ["T4"]
    for denied_gpu in deny_list:
        if any(denied_gpu in gpu.upper() for gpu in get_gpu_names()):
            return False
    return True


def get_gpu_names():
    """Detects and returns the names of available GPUs based on the backend.

    Note:
        The format and content of the returned GPU names are **not normalized**
        and vary significantly depending on the active backend. This function
        provides the names as reported by the respective backend's API."
    """
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        devices = jax.devices()

        return [getattr(d, "device_kind", "") for d in devices]

    elif backend == "tensorflow":
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return [
            tf.config.experimental.get_device_details(gpu)["device_name"]
            for gpu in gpus
        ]
    elif backend == "torch":
        import torch

        return [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    else:
        return [""]


def sharded_weights_available():
    """Whether sharded weights serialization is available.

    Returns:
        `True` if sharded weights are available, `False` otherwise.
    """
    save_weights_signature = inspect.signature(keras.saving.save_weights)
    return "max_shard_size" in save_weights_signature.parameters
