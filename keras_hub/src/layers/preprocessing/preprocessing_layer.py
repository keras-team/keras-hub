import keras

from keras_hub.src.utils.tensor_utils import assert_tf_libs_installed


class PreprocessingLayer(keras.layers.Layer):
    """Preprocessing layer base class."""

    def __init__(self, **kwargs):
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", False)
        if not _allow_python_workflow:
            assert_tf_libs_installed(self.__class__.__name__)
        super().__init__(**kwargs)
        # Don't convert inputs (we want tf tensors not backend tensors).
        self._convert_input_args = False
        # Allow raw inputs like python strings.
        self._allow_non_tensor_positional_args = True
        # Allow Python workflow. Historically, KerasHub preprocessing layers
        # required TF and TF text libraries.
        self._allow_python_workflow = _allow_python_workflow
        # Most pre-preprocessing has no build.
        if not hasattr(self, "build"):
            self.built = True

    def get_build_config(self):
        return None
