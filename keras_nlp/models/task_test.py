# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.backend import keras
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.models.task import Task
from keras_nlp.tests.test_case import TestCase
from keras_nlp.tokenizers.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    def vocabulary_size(self):
        return 10


class SimplePreprocessor(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = SimpleTokenizer()


class SimpleTask(Task):
    def __init__(self, preprocessor=None, activation=None, **kwargs):
        self.preprocessor = preprocessor
        self.activation = keras.activations.get(activation)
        inputs = keras.Input(shape=(5,))
        outputs = keras.layers.Dense(5)(inputs)
        super().__init__(inputs, outputs, **kwargs)


class TestTask(TestCase):
    def test_summary_with_preprocessor(self):
        preprocessor = SimplePreprocessor()
        model = SimpleTask(preprocessor)
        summary = []
        model.summary(print_fn=lambda x, line_break: summary.append(x))
        self.assertRegex("\n".join(summary), "Preprocessor:")

    def test_summary_without_preprocessor(self):
        model = SimpleTask()
        summary = []
        model.summary(print_fn=lambda x, line_break: summary.append(x))
        self.assertNotRegex("\n".join(summary), "Preprocessor:")

    def test_mismatched_loss(self):
        # Logit output.
        model = SimpleTask(activation=None)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        # Non-standard losses should not throw.
        model.compile(loss="mean_squared_error")
        with self.assertRaises(ValueError):
            model.compile(loss="sparse_categorical_crossentropy")
        with self.assertRaises(ValueError):
            model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False
                )
            )

        # Probability output.
        model = SimpleTask(activation="softmax")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        )
        model.compile(loss="sparse_categorical_crossentropy")
        # Non-standard losses should not throw.
        model.compile(loss="mean_squared_error")
        with self.assertRaises(ValueError):
            model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            )

        # Non-standard activations should not throw.
        model = SimpleTask(activation="tanh")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        )
