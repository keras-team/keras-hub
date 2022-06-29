# # Copyright 2022 The KerasNLP Authors
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Tests for Random Word Deletion Layer."""

# import tensorflow as tf

# from keras_nlp.layers import random_insertion


# class RandomInsertionTest(tf.test.TestCase):
#     def test_shape_and_output_from_word_insertion(self):
#         def replace_word(word):
#             replace_dict = {
#                 "Hey": "Yo",
#                 "bye": "ciao"
#             }
#             if isinstance(word, bytes):
#                 word = word.decode()
#             if word in replace_dict.keys():
#                 return replace_dict[word]
#             else:
#                 return word
#         inputs = tf.strings.split(["Hey I like", "bye bye"])
#         aug = random_insertion.RandomInsertion(1, 5, insertion_numpy_fn = replace_word, seed = 42)
#         outputs = aug(inputs)
#         outputs = tf.strings.reduce_join(outputs, separator=" ", axis=-1)
#         self.assertAllEqual(outputs.shape, inputs.shape)

#     def test_get_config_and_from_config(self):

#         augmenter = random_word_deletion.RandomWordDeletion(
#             probability=0.5, max_deletions=3
#         )

#         expected_config_subset = {"probability": 0.5, "max_deletions": 3}

#         config = augmenter.get_config()

#         self.assertEqual(config, {**config, **expected_config_subset})

#         restored_augmenter = (
#             random_word_deletion.RandomWordDeletion.from_config(
#                 config,
#             )
#         )

#         self.assertEqual(
#             restored_augmenter.get_config(),
#             {**config, **expected_config_subset},
#         )

#     def test_augment_first_batch_second(self):
#         tf.random.get_global_generator().reset_from_seed(30)
#         tf.random.set_seed(30)
#         augmenter = random_word_deletion.RandomWordDeletion(
#             probability=0.5, max_deletions=3
#         )

#         ds = tf.data.Dataset.from_tensor_slices(
#             ["samurai or ninja", "keras is good", "tensorflow is a library"]
#         )
#         ds = ds.map(augmenter)
#         ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
#         output = ds.take(1).get_single_element()

#         exp_output = [b"samurai", b"is good", b"tensorflow a library"]
#         for i in range(output.shape[0]):
#             self.assertAllEqual(output[i], exp_output[i])

#     def test_batch_first_augment_second(self):
#         tf.random.get_global_generator().reset_from_seed(30)
#         tf.random.set_seed(30)
#         augmenter = random_word_deletion.RandomWordDeletion(
#             probability=0.5, max_deletions=3
#         )

#         ds = tf.data.Dataset.from_tensor_slices(
#             ["samurai or ninja", "keras is good", "tensorflow is a library"]
#         )
#         ds = ds.batch(3).map(augmenter)
#         output = ds.take(1).get_single_element()

#         exp_output = [b"samurai", b"is good", b"tensorflow"]

#         for i in range(output.shape[0]):
#             self.assertAllEqual(output[i], exp_output[i])