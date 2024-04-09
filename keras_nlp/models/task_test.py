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

import pytest

from keras_nlp.backend import keras
from keras_nlp.models.bert.bert_classifier import BertClassifier
from keras_nlp.models.gpt2.gpt2_causal_lm import GPT2CausalLM
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
    def test_preset_accessors(self):
        bert_presets = set(BertClassifier.presets.keys())
        gpt2_presets = set(GPT2CausalLM.presets.keys())
        all_presets = set(Task.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on a task directly (it is ambiguous).
            Task.from_preset("bert_tiny_en_uncased", load_weights=False)
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertClassifier.from_preset("gpt2_base_en", load_weights=False)

    def test_summary_with_preprocessor(self):
        preprocessor = SimplePreprocessor()
        model = SimpleTask(preprocessor)
        summary = []
        model.summary(print_fn=lambda x, line_break=False: summary.append(x))
        self.assertRegex("\n".join(summary), "Preprocessor:")

    def test_summary_without_preprocessor(self):
        model = SimpleTask()
        summary = []
        model.summary(print_fn=lambda x, line_break=False: summary.append(x))
        self.assertNotRegex("\n".join(summary), "Preprocessor:")
