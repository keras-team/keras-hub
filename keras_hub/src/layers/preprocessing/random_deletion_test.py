import keras
import tensorflow as tf

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
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        augmented = augmenter(split)
        output = [
            tf.strings.reduce_join(x, separator=" ", axis=-1) for x in augmented
        ]
        exp_output = ["I like", "and Tensorflow"]
        self.assertAllEqual(output, exp_output)

    def test_shape_and_output_from_character_swaps(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        augmented = augmenter(split)
        output = [tf.strings.reduce_join(x, axis=-1) for x in augmented]
        exp_output = ["Hey I lie", "Keras and Tensoflow"]
        self.assertAllEqual(output, exp_output)

    def test_with_integer_tokens(self):
        keras.utils.set_random_seed(1337)
        inputs = tf.constant([[1, 2], [3, 4]])
        augmenter = RandomDeletion(rate=0.4, max_deletions=4, seed=42)
        output = augmenter(inputs)
        exp_output = [[2], [4]]
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
