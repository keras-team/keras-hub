import types

from keras.saving import register_keras_serializable

try:
    import namex
except ImportError:
    namex = None


def maybe_register_serializable(path, symbol):
    if isinstance(path, (list, tuple)):
        # If we have multiple export names, actually make sure to register these
        # first. This makes sure we have a backward compat mapping of old
        # serialized names to new class.
        for name in path:
            name = name.split(".")[-1]
            register_keras_serializable(package="keras_nlp", name=name)(symbol)
            register_keras_serializable(package="keras_hub", name=name)(symbol)
    if isinstance(symbol, types.FunctionType) or hasattr(symbol, "get_config"):
        # We register twice, first with keras_nlp, second with keras_hub,
        # so loading still works for classes saved as "keras_nlp".
        register_keras_serializable(package="keras_nlp")(symbol)
        register_keras_serializable(package="keras_hub")(symbol)


if namex:

    class keras_hub_export(namex.export):
        def __init__(self, path):
            super().__init__(package="keras_hub", path=path)

        def __call__(self, symbol):
            maybe_register_serializable(self.path, symbol)
            return super().__call__(symbol)

else:

    class keras_hub_export:
        def __init__(self, path):
            self.path = path

        def __call__(self, symbol):
            maybe_register_serializable(self.path, symbol)
            return symbol
