import inspect

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
        # Whether `call` has the `(x, y, sample_weight)` signature. Keras has
        # already inspected `self.call` in `Layer.__init__`, so this is safe.
        params = inspect.signature(self.call).parameters
        self._call_accepts_labels = all(
            k in params for k in ("x", "y", "sample_weight")
        )
        # Most pre-preprocessing has no build.
        if not hasattr(self, "build"):
            self.built = True

    def __call__(self, *args, **kwargs):
        # Mimic the `tf.data` behavior of unpacking `(x, y)` and
        # `(x, y, sample_weight)` tuples when calling layers that accept
        # labels. This allows layers to be mapped directly over datasets whose
        # elements are tuples, e.g. `grain.MapDataset.source(...).map(layer)`,
        # or to be called directly on a single dataset element.
        if (
            len(args) == 1
            and type(args[0]) is tuple
            and len(args[0]) in (2, 3)
            and "y" not in kwargs
            and "sample_weight" not in kwargs
            and self._call_accepts_labels
        ):
            args = args[0]
        return super().__call__(*args, **kwargs)

    def get_build_config(self):
        return None
