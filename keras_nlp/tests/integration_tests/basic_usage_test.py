# Copyright 2021 The KerasNLP Authors
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

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

import keras_nlp


class BasicUsageTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_quick_start(self, jit_compile):
        """This matches the quick start example in our base README."""

        # Tokenize some inputs with a binary label.
        vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        sentences = ["The quick brown fox jumped.", "The fox slept."]
        tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=vocab,
            sequence_length=10,
        )
        x, y = tokenizer(sentences), tf.constant([1, 0])

        # Create a tiny transformer.
        inputs = keras.Input(shape=(None,), dtype="int32")
        outputs = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=len(vocab),
            sequence_length=10,
            embedding_dim=16,
        )(inputs)
        outputs = keras_nlp.layers.TransformerEncoder(
            num_heads=4,
            intermediate_dim=32,
        )(outputs)
        outputs = keras.layers.GlobalAveragePooling1D()(outputs)
        outputs = keras.layers.Dense(1, activation="sigmoid")(outputs)
        model = keras.Model(inputs, outputs)

        # Run a single batch of gradient descent.
        model.compile(loss="binary_crossentropy", jit_compile=jit_compile)
        loss = model.train_on_batch(x, y)

        # Make sure we have a valid loss.
        self.assertGreater(loss, 0)
