import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.random_utils import record_rng
from keras_hub.src.utils.tensor_utils import canonicalize_python_inputs
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.layers.RandomDeletion")
class RandomDeletion(PreprocessingLayer):
    """Augments input by randomly deleting tokens.

    This layer comes in handy when you need to generate new data using deletion
    augmentation as described in the paper [EDA: Easy Data Augmentation
    Techniques for Boosting Performance on Text Classification Tasks]
    (https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
    pre-split into token level inputs. This allows control over the level of
    augmentation, you can split by character for character level swaps, or by
    word for word level swaps.

    Input data should be passed as tensors, `tf.RaggedTensor`s, or lists. For
    batched input, inputs should be a list of lists or a rank two tensor. For
    unbatched inputs, each element should be a list or a rank one tensor.

    This layer runs on a pure Python/NumPy code path by default, so it works
    inside a Grain pipeline and does not require TensorFlow. Randomness is
    derived from `seed` together with a stable hash of the record being
    augmented, so the output does not depend on how many Grain workers are
    running or on the order records are seen in. The tradeoff is that an
    identical record is augmented identically on every epoch. To vary the
    augmentation per epoch, pass your own generator as `rng`, for example from
    `grain.RandomMapTransform`, which derives one from the element index.

    Args:
        rate: The probability of a token being chosen for deletion.
        max_deletions: The maximum number of tokens to delete.
        skip_list: A list of token values that should not be considered
            candidates for deletion.
        skip_fn: A function that takes as input a scalar tensor token and
            returns as output a scalar tensor True/False value. A value of
            True indicates that the token should not be considered a
            candidate for deletion. This function must be tracable--it
            should consist of tensorflow operations. Setting this forces the
            layer onto the TensorFlow code path, which cannot run in a Grain
            worker; prefer `skip_py_fn`.
        skip_py_fn: A function that takes as input a python token value and
            returns as output `True` or `False`. A value of True
            indicates that should not be considered a candidate for deletion.
            Unlike the `skip_fn` argument, this argument need not be
            tracable--it can be any python function.
        seed: A seed for the random number generator.

    Call arguments:
        inputs: The tokens to augment.
        rng: Optional `np.random.Generator` used instead of the per record
            generator derived from `seed`. Only supported on the pure Python
            code path.

    Examples:

    Word level usage.
    >>> keras.utils.set_random_seed(1337)
    >>> x = ["Hey I like", "Keras and Tensorflow"]
    >>> x = list(map(lambda x: x.split(), x))
    >>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4, seed=1)
    >>> y = augmenter(x)
    >>> list(map(lambda y: " ".join(y), y))
    ['Hey like', 'and Tensorflow']

    Character level usage.
    >>> keras.utils.set_random_seed(1337)
    >>> x = ["Hey Dude", "Speed Up"]
    >>> x = list(map(lambda x: list(x), x))
    >>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4, seed=3)
    >>> y = augmenter(x)
    >>> list(map(lambda y: "".join(y), y))
    ['HeyDue', 'ee Up']

    Usage with skip_list.
    >>> keras.utils.set_random_seed(1337)
    >>> x = ["Hey I like", "Keras and Tensorflow"]
    >>> x = list(map(lambda x: x.split(), x))
    >>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4,
    ...     skip_list=["Keras", "Tensorflow"], seed=9)
    >>> y = augmenter(x)
    >>> list(map(lambda y: " ".join(y), y))
    ['I like', 'Keras Tensorflow']

    Usage with skip_fn.
    >>> def skip_fn(word):
    ...     return tf.strings.regex_full_match(word, r"\\pP")
    >>> keras.utils.set_random_seed(1337)
    >>> x = ["Hey I like", "Keras and Tensorflow"]
    >>> x = list(map(lambda x: x.split(), x))
    >>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4,
    ...     skip_fn=skip_fn, seed=42)
    >>> y = augmenter(x)
    >>> list(map(lambda y: " ".join(y), y))
    ['I like', 'and']

    Usage with skip_py_fn.
    >>> def skip_py_fn(word):
    ...     return len(word) < 4
    >>> keras.utils.set_random_seed(1337)
    >>> x = ["Hey I like", "Keras and Tensorflow"]
    >>> x = list(map(lambda x: x.split(), x))
    >>> augmenter = RandomDeletion(rate=0.4,
    ...     skip_py_fn=skip_py_fn, seed=14)
    >>> y = augmenter(x)
    >>> list(map(lambda y: " ".join(y), y))
    ['Hey I', 'and Tensorflow']
    """

    def __init__(
        self,
        rate,
        max_deletions=None,
        skip_list=None,
        skip_fn=None,
        skip_py_fn=None,
        seed=None,
        name=None,
        dtype="int32",
        **kwargs,
    ):
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            dtype=dtype,
            name=name,
            _allow_python_workflow=_allow_python_workflow,
            **kwargs,
        )

        self.rate = rate
        self.max_deletions = max_deletions
        self.seed = random.randint(1, int(1e9)) if seed is None else seed
        # `tf.random.Generator` is only built on demand by `_call_tf`. Building
        # it here would require TensorFlow at construction time, and the
        # generator would be pickled into every Grain worker, making all
        # workers replay the same stream.
        self._generator = None
        self.skip_list = skip_list
        self.skip_fn = skip_fn
        self.skip_py_fn = skip_py_fn
        if self.max_deletions is not None and self.max_deletions < 0:
            raise ValueError(
                "max_deletions must be non-negative."
                f"Received max_deletions={max_deletions}."
            )

        if self.rate > 1 or self.rate < 0:
            raise ValueError(
                "Rate must be between 0 and 1 (both inclusive)."
                f"Received: rate={rate}"
            )

        if [self.skip_list, self.skip_fn, self.skip_py_fn].count(None) < 2:
            raise ValueError(
                "Exactly one of `skip_list`, `skip_fn`, `skip_py_fn` must be "
                "provided."
            )

        self._skip_set = set(self.skip_list) if self.skip_list else None
        # Built on demand by `_call_tf`; see the `_generator` comment above.
        self._skip_table = None

    def _tf_generator(self):
        if self._generator is None:
            self._generator = tf.random.Generator.from_seed(self.seed)
        return self._generator

    def _tf_skip_table(self):
        if self._skip_table is None:
            self._skip_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.convert_to_tensor(self.skip_list),
                    tf.convert_to_tensor([True] * len(self.skip_list)),
                ),
                default_value=False,
            )
        return self._skip_table

    @preprocessing_function
    def _call_tf(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)

        skip_masks = None
        if self.skip_list:
            skip_masks = self._tf_skip_table().lookup(inputs.flat_values)
        elif self.skip_fn:
            skip_masks = tf.map_fn(
                self.skip_fn, inputs.flat_values, fn_output_signature="bool"
            )
        elif self.skip_py_fn:

            def string_fn(token):
                return self.skip_py_fn(token.numpy().decode("utf-8"))

            def int_fn(token):
                return self.skip_py_fn(token.numpy())

            py_fn = string_fn if inputs.dtype == tf.string else int_fn

            skip_masks = tf.map_fn(
                lambda x: tf.py_function(py_fn, [x], "bool"),
                inputs.flat_values,
                fn_output_signature="bool",
            )

        positions_flat = tf.range(tf.size(inputs.flat_values))
        positions = inputs.with_flat_values(positions_flat)
        if skip_masks is not None:
            skip_masks = tf.logical_not(skip_masks)
            skip_masks.set_shape([None])
            positions = tf.ragged.boolean_mask(
                positions, inputs.with_flat_values(skip_masks)
            )

        # Figure out how many we are going to select.
        token_counts = tf.cast(positions.row_lengths(), "float32")
        num_to_select = tf.random.stateless_binomial(
            shape=tf.shape(token_counts),
            seed=self._tf_generator().make_seeds()[:, 0],
            counts=token_counts,
            probs=self.rate,
        )
        if self.max_deletions is not None:
            num_to_select = tf.math.minimum(num_to_select, self.max_deletions)
        num_to_select = tf.cast(num_to_select, "int64")

        # Shuffle and trim to items that are going to be selected.
        def _shuffle_and_trim(x):
            positions, top_n = x
            shuffled = tf.random.shuffle(positions, seed=self.seed)
            return shuffled[:top_n]

        selected_for_mask = tf.map_fn(
            _shuffle_and_trim,
            (positions, num_to_select),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype
            ),
        )
        selected_for_mask.flat_values.set_shape([None])

        # Construct the mask which is a boolean RT
        # Scatter 0's to positions that have been selector for deletion.
        update_values = tf.zeros_like(selected_for_mask.flat_values, "int32")
        update_indices = selected_for_mask.flat_values
        update_indices = tf.expand_dims(update_indices, -1)
        update_indices = tf.cast(update_indices, "int32")
        mask_flat = tf.ones_like(inputs.flat_values, dtype="int32")
        mask_flat = tf.tensor_scatter_nd_update(
            mask_flat, update_indices, update_values
        )
        mask = tf.cast(inputs.with_flat_values(mask_flat), "bool")

        inputs = tf.ragged.boolean_mask(inputs, mask)

        if unbatched:
            inputs = tf.squeeze(inputs, axis=0)

        return inputs

    def _skip_mask_python(self, tokens):
        """Return a bool per token, `True` if it cannot be deleted."""
        if self._skip_set is not None:
            return [token in self._skip_set for token in tokens]
        if self.skip_py_fn is not None:
            return [bool(self.skip_py_fn(token)) for token in tokens]
        return [False] * len(tokens)

    def _call_python(self, inputs, rng=None):
        inputs, batched = canonicalize_python_inputs(inputs)

        outputs = []
        for row in inputs:
            # `tf.RaggedTensor.to_list()` yields `bytes` for string data.
            row = [
                token.decode("utf-8") if isinstance(token, bytes) else token
                for token in row
            ]
            row_rng = rng if rng is not None else record_rng(self.seed, row)
            candidates = [
                index
                for index, skip in enumerate(self._skip_mask_python(row))
                if not skip
            ]
            num_to_select = int(row_rng.binomial(len(candidates), self.rate))
            if self.max_deletions is not None:
                num_to_select = min(num_to_select, self.max_deletions)
            deleted = set()
            if num_to_select:
                shuffled = row_rng.permutation(candidates)
                deleted = set(shuffled[:num_to_select].tolist())
            outputs.append(
                [token for i, token in enumerate(row) if i not in deleted]
            )

        if not batched:
            outputs = outputs[0]
        # Deletion always produces ragged output, which the TensorFlow path
        # also returns as nested python lists.
        return outputs

    def call(self, inputs, rng=None):
        # `skip_fn` is documented to consist of TensorFlow ops, so a layer
        # configured with it can only run on the TensorFlow path.
        use_tf = (
            not self._allow_python_workflow
            or self.skip_fn is not None
            or in_tf_function()
        )
        if use_tf:
            if rng is not None:
                raise ValueError(
                    "`rng` is only supported on the pure python code path, "
                    "but this layer is running on the TensorFlow code path. "
                    "This happens when `skip_fn` is set or when the layer is "
                    "called inside a `tf.function` such as `tf.data.Dataset."
                    "map`. Use `skip_py_fn` instead of `skip_fn`, and Grain "
                    "instead of `tf.data`, to pass an `rng`."
                )
            return self._call_tf(inputs)
        return self._call_python(inputs, rng=rng)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "max_deletions": self.max_deletions,
                "seed": self.seed,
                "skip_list": self.skip_list,
                "skip_fn": self.skip_fn,
                "skip_py_fn": self.skip_py_fn,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = None
        return tuple(inputs_shape)
