import keras

from keras_hub.src.utils.tensor_utils import assert_tf_libs_installed


class PreprocessingLayer(keras.layers.Layer):
    """Preprocessing layer base class."""

    def __init__(self, **kwargs):
        assert_tf_libs_installed(self.__class__.__name__)
        super().__init__(**kwargs)
        # Don't convert inputs (we want tf tensors not backend tensors).
        self._convert_input_args = False
        # Allow raw inputs like python strings.
        self._allow_non_tensor_positional_args = True
        # Most pre-preprocessing has no build.
        if not hasattr(self, "build"):
            self.built = True

    def get_build_config(self):
        return None
