import keras


class PreprocessingLayer(keras.layers.Layer):
    """Preprocessing layer base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Don't convert inputs.
        self._convert_input_args = False
        # Allow raw inputs like python strings.
        self._allow_non_tensor_positional_args = True
        # Most pre-preprocessing has no build.
        if not hasattr(self, "build"):
            self.built = True

    def get_build_config(self):
        return None
