import inspect

import keras

from keras_hub.src.utils.tensor_utils import assert_tf_libs_installed
from keras_hub.src.utils.tensor_utils import in_tf_function


class PreprocessingLayer(keras.layers.Layer):
    """Preprocessing layer base class."""

    def __init__(self, **kwargs):
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(**kwargs)
        # Don't convert inputs (we want tf tensors not backend tensors).
        self._convert_input_args = False
        # Allow raw inputs like python strings.
        self._allow_non_tensor_positional_args = True
        # Whether eager calls run the pure Python path (the default) or the
        # TensorFlow path. Historically, KerasHub preprocessing layers required
        # TF and TF text libraries. Now they are only required when tracing a
        # `tf.function` or `tf.data` pipeline, or when this is set to `False`,
        # and the check happens on the first TF path use rather than here. See
        # `_use_tf_workflow()`.
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

    def _use_tf_workflow(self):
        """Whether to run the TensorFlow path instead of the pure Python path.

        The TensorFlow path is always used when tracing a `tf.function` or a
        `tf.data` pipeline. Outside of that, it is only used when the layer was
        constructed with `_allow_python_workflow=False`, in which case
        TensorFlow and TensorFlow Text must be installed. This is where that
        requirement is checked, so that layers can be constructed (and run on
        the Python path, e.g. inside a Grain pipeline) without TensorFlow.

        The flag is per layer and is not inherited: a model preprocessor's
        tokenizer, packers and converters each carry their own, so passing
        `_allow_python_workflow=False` to a composite preprocessor only affects
        its own `call`, not the layers it delegates to.
        """
        if in_tf_function():
            return True
        if self._allow_python_workflow:
            return False
        assert_tf_libs_installed(self.__class__.__name__)
        return True

    def get_build_config(self):
        return None
