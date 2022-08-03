# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import tensorflow as tf
from tensorflow import keras


class RandomReplacement(keras.layers.Layer):
    """Augments input by randomly replacing words.

    Args:
        rate: A float in [0, 1] that is the rate of replacement
        max_replacements: An integer that is the maximum number of replacements
        replacement_list: list of candidates to uniformly sample form to replace
        replacement_fn: function that takes in a token and returns a replacement
        token
        py_replacement_fn: python version of replacement_fn.
        skip_list: list of tokens to skip.
        skip_fn: fn that takes in a token and returns a boolean.
        py_skip_fn: python numpy version of skip_fn.
        seed: An integer that is the seed for the random number generator.

    Examples:

    Word level usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.3, max_replacements=2, seed=42,
    ... replacement_list=['Random1', 'Random2', 'Random3'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy= array([b'Random1 Random2 like', b'Random1 Random2 Tensorflow'],
    dtype=object)>

    Character level usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.unicode_split(["Hey Dude", "Speed Up"], "UTF-8")
    >>> augmenter=RandomReplacement(rate=0.3, max_replacements=2, seed=42,
    ... replacement_list=['x', 'y', 'z'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'HexyDude', b'Spxed Uy'], dtype=object)>

    Usage with replacement_fn
    >>> def replacement_fn(word):
    ...   if word == 'Keras':
    ...     return 'KerasNLP'
    ...   return word
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.7, max_replacements=6, seed=42,
    ... replacement_fn=replacement_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I like', b'Keras and Tensorflow'], dtype=object)>

    Usage with py_replacement_fn
    >>> def py_replacement_fn(word):
    ...   if word in ['Keras', 'TensorFlow', 'like']:
    ...     return 'KerasNLP'
    ...   return word
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.5, max_replacements=6, seed=42,
    ... py_replacement_fn=py_replacement_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I KerasNLP', b'KerasNLP and Tensorflow'], dtype=object)>

    Usage with skip_list
    >>> def py_replacement_fn(word):
    ...   if word in ['Keras', 'TensorFlow', 'like']:
    ...     return 'KerasNLP'
    ...   return word
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.5, max_replacements=6, seed=42,
    ... py_replacement_fn=py_replacement_fn, skip_list=['Keras'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I KerasNLP', b'Keras and Tensorflow'], dtype=object)>

    Usage with skip_fn
    >>> def py_replacement_fn(word):
    ...   if word in ['Keras', 'TensorFlow', 'like']:
    ...     return 'KerasNLP'
    ...   return word
    >>> def skip_fn(word):
    ...   if word == 'like':
    ...     return True
    ...   return False
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.5, max_replacements=6, seed=42,
    ... py_replacement_fn=py_replacement_fn, skip_fn=skip_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I like', b'KerasNLP and Tensorflow'], dtype=object)>

    Usage with py_skip_fn
    >>> def py_replacement_fn(word):
    ...   if word in ['Keras', 'TensorFlow', 'like']:
    ...     return 'KerasNLP'
    ...   return word
    >>> def py_skip_fn(word):
    ...   if word == 'like':
    ...     return True
    ...   return False
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomReplacement(rate=0.5, max_replacements=6, seed=42,
    ... py_replacement_fn=py_replacement_fn, py_skip_fn=py_skip_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I like', b'KerasNLP and Tensorflow'], dtype=object)>
    """

    def __init__(
        self,
        rate,
        max_replacements,
        replacement_list=None,
        replacement_fn=None,
        py_replacement_fn=None,
        skip_list=None,
        skip_fn=None,
        py_skip_fn=None,
        seed=None,
        name=None,
        **kwargs,
    ):
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be an integer type or a string. "
                    f"Received: dtype={dtype}"
                )

        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.max_replacements = max_replacements
        self.replacement_list = replacement_list
        self.replacement_fn = replacement_fn
        self.py_replacement_fn = py_replacement_fn
        self.skip_list = skip_list
        self.skip_fn = skip_fn
        self.py_skip_fn = py_skip_fn
        self.seed = random.randint(1, 1e9) if seed is None else seed
        self._generator = tf.random.Generator.from_seed(self.seed)
        if self.rate > 1 or self.rate < 0:
            raise ValueError(
                "Rate must be between 0 and 1 (both inclusive)."
                f"Received: rate={rate}"
            )

        if [self.skip_list, self.skip_fn, self.py_skip_fn].count(None) < 2:
            raise ValueError(
                "Exactly one of skip_list, skip_fn, py_skip_fn must be "
                "provided."
            )

        countReplaceOptions = [
            self.replacement_list,
            self.replacement_fn,
            self.py_replacement_fn,
        ].count(None)
        if countReplaceOptions != 2:
            raise ValueError(
                "Exactly one of replacement_list, replacement_fn, "
                "py_replacement_fn must be provided."
            )

        if self.skip_list:
            self.StaticHashTable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.convert_to_tensor(self.skip_list),
                    tf.convert_to_tensor([True] * len(self.skip_list)),
                ),
                default_value=False,
            )

    @tf.function
    def call(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        input_is_1d = False
        if inputs.shape.rank < 1 or inputs.shape.rank > 2:
            raise ValueError(
                "Input must either be rank 1 or rank 2. Received input with "
                f"rank={inputs.shape.rank}"
            )
        elif inputs.shape.rank == 1:
            input_is_1d = True
            # Add a new axis at the beginning.
            inputs = tf.expand_dims(inputs, axis=0)
        if isinstance(inputs, tf.Tensor):
            # Convert to ragged tensor.
            inputs = tf.RaggedTensor.from_tensor(inputs)

        def _check_skip(token):
            if self.skip_list:
                return self.StaticHashTable.lookup(token)
            elif self.skip_fn:
                return self.skip_fn(token)
            elif self.py_skip_fn:

                def _preprocess_skip_fn(word):
                    return self.py_skip_fn(word.numpy().decode("utf-8"))

                return tf.py_function(_preprocess_skip_fn, [token], tf.bool)
            else:
                return False

        # Figure out how many we are going to select.
        token_counts = tf.cast(inputs.row_lengths(), "float32")
        num_to_select = tf.random.stateless_binomial(
            shape=tf.shape(token_counts),
            seed=self._generator.make_seeds()[:, 0],
            counts=token_counts,
            probs=self.rate,
        )
        if self.max_replacements is not None:
            num_to_select = tf.math.minimum(num_to_select, self.max_replacements)
        num_to_select = tf.cast(num_to_select, "int64")

        def _replace(x):
            """
            Replace words randomly
            """
            inputs, num_to_select = x
            for _ in range(num_to_select):
                # Choose a Random Index
                index = tf.random.stateless_uniform(
                    shape=[], minval=0, maxval=tf.size(inputs), dtype=tf.int32, 
                    seed=self._generator.make_seeds()[:, 0],
                )
                synonym = inputs[index]
                if _check_skip(synonym):
                    continue
                if self.replacement_fn is not None:
                    self.replacement_fn(synonym)
                    inputs = tf.tensor_scatter_nd_update(
                        inputs, [[index]], [synonym]
                    )
                elif self.py_replacement_fn is not None:

                    def _preprocess_replace_fn(word):
                        return self.py_replacement_fn(
                            word.numpy().decode("utf-8")
                        )

                    synonym = tf.py_function(
                        _preprocess_replace_fn, [synonym], tf.string
                    )
                    inputs = tf.tensor_scatter_nd_update(
                        inputs, [[index]], [synonym]
                    )
                elif self.replacement_list is not None:
                    replace_list_index = tf.random.stateless_uniform(
                        shape=[], minval=0, maxval=len(self.replacement_list), 
                        dtype=tf.int32, seed=self._generator.make_seeds()[:, 0],
                    )
                    synonym = tf.gather(
                        self.replacement_list, replace_list_index
                    )
                    inputs = tf.tensor_scatter_nd_update(
                        inputs, [[index]], [synonym]
                    )
            return inputs

        replaced = tf.map_fn(
            _replace,
            (inputs, num_to_select),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=inputs.ragged_rank - 1, dtype=inputs.dtype
            ),
        )
        replaced.flat_values.set_shape([None])

        if input_is_1d:
            replaced = tf.squeeze(replaced, axis=0)
        return replaced

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "max_replacements": self.max_replacements,
                "seed": self.seed,
                "skip_list": self.skip_list,
                "skip_fn": self.skip_fn,
                "skip_py_fn": self.skip_py_fn,
                "replacement_list": self.replacement_list,
                "replacement_fn": self.replacement_fn,
                "py_replacement_fn": self.py_replacement_fn,
            }
        )
        return config