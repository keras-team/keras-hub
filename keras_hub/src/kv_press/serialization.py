import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.press import KVCachePress
from keras_hub.src.kv_press.random_press import RandomPress
from keras_hub.src.kv_press.streaming_llm_press import StreamingLLMPress


@keras_hub_export("keras_hub.press.serialize")
def serialize(press):
    return keras.saving.serialize_keras_object(press)


@keras_hub_export("keras_hub.press.deserialize")
def deserialize(config, custom_objects=None):
    """Return a `KVCachePress` object from its config."""
    all_classes = {
        "knorm": KnormPress,
        "random": RandomPress,
        "streaming_llm": StreamingLLMPress,
    }
    return keras.saving.deserialize_keras_object(
        config,
        module_objects=all_classes,
        custom_objects=custom_objects,
        printable_module_name="kv cache presses",
    )


@keras_hub_export("keras_hub.press.get")
def get(identifier):
    """Retrieve a KerasHub KV cache press by the identifier.

    The `identifier` may be the string name of a press class, a config
    dict, an already-constructed `KVCachePress` instance, or `None`.

    >>> identifier = 'knorm'
    >>> press = keras_hub.press.get(identifier)

    You can also specify `config` of the press to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note
    that the `class_name` must map to a `KVCachePress` class.

    >>> cfg = {'class_name': 'keras_hub>KnormPress', 'config': {}}
    >>> press = keras_hub.press.get(cfg)

    Args:
        identifier: String, dict, `KVCachePress` instance, or `None`.

    Returns:
        `KVCachePress` instance (or `None`) based on the input identifier.

    Raises:
        ValueError: If the input identifier is not a supported type or in a
            bad format.
    """
    if identifier is None:
        return None
    if isinstance(identifier, KVCachePress):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        if not identifier.islower():
            raise KeyError(
                "`keras_hub.press.get()` must take a lowercase string "
                f"identifier, but received: {identifier}."
            )
        return deserialize(identifier)
    raise ValueError(
        "Could not interpret kv cache press identifier: " + str(identifier)
    )
