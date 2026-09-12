import sys

import grain
import keras
import numpy as np
import pytest
import tensorflow as tf
from absl import flags

from keras_hub.src.layers.preprocessing.random_deletion import RandomDeletion
from keras_hub.src.tests.test_case import TestCase


class RandomDeletionTest(TestCase):
    def test_layer_basics(self):
        # `rate=1.0` always deletes every token, so the result doesn't
        # depend on the random draw. Int dtype avoids a numpy
        # dtype-promotion crash in `assertAllClose` on string RaggedTensors.
        keras.utils.set_random_seed(1337)
        self.run_preprocessing_layer_test(
            cls=RandomDeletion,
            init_kwargs={"rate": 1.0},
            input_data=tf.constant([[1, 2], [3, 4]]),
            expected_output=[[], []],
        )

    def test_shape_and_output_from_word_deletion(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=1)
        augmented = augmenter(split)
        output = [
            tf.strings.reduce_join(x, separator=" ", axis=-1) for x in augmented
        ]
        exp_output = ["Hey like", "and Tensorflow"]
        self.assertAllEqual(output, exp_output)
        # `max_deletions=1`, so each row loses at most one token.
        for row, augmented_row in zip(split.to_list(), augmented):
            self.assertLen(augmented_row, len(row) - 1)

    def test_shape_and_output_from_character_swaps(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=1)
        augmented = augmenter(split)
        output = [tf.strings.reduce_join(x, axis=-1) for x in augmented]
        exp_output = ["He I like", "Keras and Tensorflw"]
        self.assertAllEqual(output, exp_output)

    def test_with_integer_tokens(self):
        inputs = tf.constant([[1, 2], [3, 4]])
        augmenter = RandomDeletion(rate=0.4, max_deletions=4, seed=13)
        output = augmenter(inputs)
        exp_output = [[2], [3]]
        self.assertAllEqual(output, exp_output)

    def test_skip_options(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomDeletion(
            rate=1.0, max_deletions=2, skip_list=["Tensorflow", "like"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        exp_output = ["like", "Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.equal(word, "Tensorflow") or tf.equal(word, "like")

        augmenter = RandomDeletion(rate=1.0, max_deletions=2, skip_fn=skip_fn)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output, exp_output)

        def skip_py_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomDeletion(
            rate=1.0, max_deletions=2, skip_py_fn=skip_py_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output, exp_output)

    def test_augment_first_batch_second(self):
        # Only skip_fn/skip_py_fn are covered; the no-skip path is already
        # covered by test_layer_basics.
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_fn=skip_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [["I", "like"], ["and", "Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [["Hey", "I", "like"], ["and", "Tensorflow"]]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_augment_second(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_fn=skip_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [["I", "like"], ["and", "Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [["Hey", "I", "like"], ["and", "Tensorflow"]]
        self.assertAllEqual(output, exp_output)

    def test_grain_outputs_python_lists(self):
        augmenter = RandomDeletion(rate=0.4, seed=42)
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
        augmenter = RandomDeletion(rate=0.5, seed=42)
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
        augmenter = RandomDeletion(rate=0.5, seed=42)
        data = [[f"tok{i}", "a", "b", "c", "d"] for i in range(16)]
        expected = [augmenter(row) for row in data]
        ds = grain.MapDataset.source(data).map(augmenter).to_iter_dataset()
        for num_workers in (1, 2):
            options = grain.MultiprocessingOptions(num_workers=num_workers)
            self.assertEqual(list(ds.mp_prefetch(options)), expected)

    def test_rng_argument(self):
        augmenter = RandomDeletion(rate=0.5, seed=42)
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

        augmenter = RandomDeletion(rate=0.5, seed=42, skip_fn=skip_fn)
        with self.assertRaisesRegex(ValueError, "`rng` is only supported"):
            augmenter(["a", "b"], rng=np.random.default_rng(0))

    def test_python_and_tf_paths_agree_on_structure(self):
        inputs = [["Hey", "I", "like"], ["Keras", "and", "Tensorflow"]]
        python_layer = RandomDeletion(rate=0.5, max_deletions=1, seed=42)
        tf_layer = RandomDeletion(
            rate=0.5, max_deletions=1, seed=42, _allow_python_workflow=False
        )
        python_output = python_layer(inputs)
        tf_output = tf_layer(inputs)
        self.assertLen(python_output, len(tf_output))
        for row, python_row, tf_row in zip(inputs, python_output, tf_output):
            # The two paths draw from different RNG streams, but both must
            # delete at most `max_deletions` tokens and keep the original
            # order of whatever survives.
            for out_row in (python_row, tf_row):
                self.assertGreaterEqual(len(out_row), len(row) - 1)
                self.assertEqual(out_row, [t for t in row if t in out_row])
