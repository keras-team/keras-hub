import sys

import grain
import keras
import numpy as np
import pytest
import tensorflow as tf
from absl import flags

from keras_hub.src.layers.preprocessing.random_swap import RandomSwap
from keras_hub.src.tests.test_case import TestCase


class RandomSwapTest(TestCase):
    def test_layer_basics(self):
        # `rate=0.0` never swaps, so the result doesn't depend on the
        # random draw. Int dtype avoids a numpy dtype-promotion crash in
        # `assertAllClose` on string RaggedTensors.
        self.run_preprocessing_layer_test(
            cls=RandomSwap,
            init_kwargs={"rate": 0.0},
            input_data=tf.constant([[1, 2, 3], [4, 5, 6]]),
            expected_output=[[1, 2, 3], [4, 5, 6]],
        )

    def test_shape_and_output_from_word_swap(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = RandomSwap(rate=0.7, max_swaps=3, seed=1)
        augmented = augmenter(split)
        output = [
            tf.strings.reduce_join(x, separator=" ", axis=-1) for x in augmented
        ]
        exp_output = ["like I Hey", "and Keras Tensorflow"]
        self.assertAllEqual(output, exp_output)
        # Swapping is a permutation, so every row keeps its exact tokens.
        for row, augmented_row in zip(split.to_list(), augmented):
            self.assertEqual(
                sorted(t.decode("utf-8") for t in row), sorted(augmented_row)
            )

    def test_shape_and_output_from_character_swap(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = RandomSwap(rate=0.7, max_swaps=6, seed=1)
        augmented = augmenter(split)
        output = [tf.strings.reduce_join(x, axis=-1) for x in augmented]
        exp_output = ["H leIkyi e", "Kerasnanf eTosor ldw"]
        self.assertAllEqual(output, exp_output)

    def test_with_integer_tokens(self):
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
        augmenter = RandomSwap(rate=0.7, max_swaps=6, seed=1)
        output = augmenter(inputs)
        exp_output = [[1, 3, 2], [5, 4, 6]]
        self.assertAllEqual(output, exp_output)

    def test_skip_options(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        # `skip_list` and `skip_py_fn` run on the python path and `skip_fn` on
        # the TensorFlow path, so they draw from different RNG streams and
        # need different seeds to produce a visible swap.
        augmenter = RandomSwap(
            rate=0.9, max_swaps=3, seed=3, skip_list=["Tensorflow", "like"]
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        exp_output = ["I Hey like", "and Keras Tensorflow"]
        self.assertAllEqual(output, exp_output)
        # Skipped tokens must stay at their original index.
        self.assertEqual(augmented[0][2], "like")
        self.assertEqual(augmented[1][2], "Tensorflow")

        def skip_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomSwap(rate=0.9, max_swaps=3, seed=11, skip_fn=skip_fn)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        exp_output = ["I Hey like", "Keras and Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_py_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomSwap(
            rate=0.9, max_swaps=3, seed=3, skip_py_fn=skip_py_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        exp_output = ["I Hey like", "and Keras Tensorflow"]
        self.assertAllEqual(output, exp_output)

    def test_augment_first_batch_second(self):
        # Only skip_fn/skip_py_fn are covered; the no-skip path is already
        # covered by test_layer_basics.
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)

        def skip_fn(word):
            # Regex to match words starting with I or a
            return tf.strings.regex_full_match(word, r"[I, a].*")

        def skip_py_fn(word):
            return len(word) < 2

        augmenter = RandomSwap(rate=0.7, max_swaps=5, seed=11, skip_fn=skip_fn)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [
            ["like", "I", "Hey"],
            ["Keras", "and", "Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomSwap(
            rate=0.7, max_swaps=2, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [
            ["Hey", "I", "like"],
            ["Tensorflow", "Keras", "and"],
        ]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_augment_second(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)

        def skip_fn(word):
            # Regex to match words starting with I
            return tf.strings.regex_full_match(word, r"[I].*")

        def skip_py_fn(word):
            return len(word) < 2

        augmenter = RandomSwap(rate=0.7, max_swaps=2, seed=42, skip_fn=skip_fn)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(2).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [
            ["Hey", "I", "like"],
            ["and", "Keras", "Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomSwap(
            rate=0.7, max_swaps=2, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(2).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [
            ["Hey", "I", "like"],
            ["and", "Keras", "Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

    def test_grain_outputs_python_lists(self):
        augmenter = RandomSwap(rate=0.4, seed=42)
        data = [["Hey", "I", "like"], ["Keras", "and", "Tensorflow"]]
        ds = grain.MapDataset.source(data).map(augmenter)
        output = list(ds)
        self.assertLen(output, 2)
        for row in output:
            self.assertIsInstance(row, list)
            for token in row:
                self.assertIsInstance(token, str)

    def test_output_independent_of_record_order(self):
        # This is the property that makes the layer safe to run with any
        # number of Grain workers: a record is augmented the same way no
        # matter when, where, or in what batch it is seen.
        augmenter = RandomSwap(rate=0.5, seed=42)
        data = [[f"tok{i}", "a", "b", "c", "d"] for i in range(8)]
        forward = [augmenter(row) for row in data]
        backward = [augmenter(row) for row in reversed(data)]
        self.assertEqual(forward, list(reversed(backward)))
        # Batched and unbatched calls agree too.
        self.assertEqual(augmenter(data), forward)

    @pytest.mark.large
    def test_grain_multiprocessing_is_deterministic(self):
        # Grain reads absl flags when it spawns workers, and pytest never
        # parses them.
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(sys.argv[:1])
        augmenter = RandomSwap(rate=0.5, seed=42)
        data = [[f"tok{i}", "a", "b", "c", "d"] for i in range(16)]
        expected = [augmenter(row) for row in data]
        ds = grain.MapDataset.source(data).map(augmenter).to_iter_dataset()
        for num_workers in (1, 2):
            options = grain.MultiprocessingOptions(num_workers=num_workers)
            self.assertEqual(list(ds.mp_prefetch(options)), expected)

    def test_rng_argument(self):
        augmenter = RandomSwap(rate=0.5, seed=42)
        row = ["a", "b", "c", "d", "e", "f", "g", "h"]
        # An explicit generator advances between calls, so augmentation
        # varies, and replaying the generator reproduces the sequence.
        rng = np.random.default_rng(0)
        first = [tuple(augmenter(row, rng=rng)) for _ in range(8)]
        rng = np.random.default_rng(0)
        second = [tuple(augmenter(row, rng=rng)) for _ in range(8)]
        self.assertGreater(len(set(first)), 1)
        self.assertEqual(first, second)

    def test_rng_rejected_on_tf_path(self):
        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        augmenter = RandomSwap(rate=0.5, seed=42, skip_fn=skip_fn)
        with self.assertRaisesRegex(ValueError, "`rng` is only supported"):
            augmenter(["a", "b"], rng=np.random.default_rng(0))

    def test_python_and_tf_paths_agree_on_structure(self):
        inputs = [["Hey", "I", "like"], ["Keras", "and", "Tensorflow"]]
        python_layer = RandomSwap(rate=0.5, max_swaps=2, seed=42)
        tf_layer = RandomSwap(
            rate=0.5, max_swaps=2, seed=42, _allow_python_workflow=False
        )
        for output in (python_layer(inputs), tf_layer(inputs)):
            self.assertLen(output, len(inputs))
            for row, out_row in zip(inputs, output):
                # The two paths draw from different RNG streams, but both
                # must permute each row without adding or losing tokens.
                self.assertEqual(sorted(out_row), sorted(row))
