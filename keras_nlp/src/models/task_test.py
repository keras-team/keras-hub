# Copyright 2024 The KerasNLP Authors
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

import os
import pathlib

import keras
import pytest

from keras_nlp.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_nlp.src.models.preprocessor import Preprocessor
from keras_nlp.src.models.task import Task
from keras_nlp.src.models.text_classifier import TextClassifier
from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.tokenizers.tokenizer import Tokenizer
from keras_nlp.src.utils.preset_utils import CONFIG_FILE
from keras_nlp.src.utils.preset_utils import METADATA_FILE
from keras_nlp.src.utils.preset_utils import MODEL_WEIGHTS_FILE
from keras_nlp.src.utils.preset_utils import TASK_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import TASK_WEIGHTS_FILE
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import load_json


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
        bert_presets = set(BertTextClassifier.presets.keys())
        gpt2_presets = set(GPT2CausalLM.presets.keys())
        all_presets = set(Task.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)
        self.assertIn("bert_tiny_en_uncased", bert_presets)
        self.assertNotIn("bert_tiny_en_uncased", gpt2_presets)
        self.assertIn("gpt2_base_en", gpt2_presets)
        self.assertNotIn("gpt2_base_en", bert_presets)
        self.assertIn("bert_tiny_en_uncased", all_presets)
        self.assertIn("gpt2_base_en", all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            CausalLM.from_preset("gpt2_base_en", load_weights=False),
            GPT2CausalLM,
        )
        # TODO: Add a classifier task loading test when there is a classifier
        # with new design available on Kaggle.

    @pytest.mark.large
    def test_from_preset_with_kwargs(self):
        # Test `dtype`
        model = CausalLM.from_preset(
            "gpt2_base_en", load_weights=False, dtype="bfloat16"
        )
        self.assertIsInstance(model, GPT2CausalLM)
        self.assertEqual(model.dtype_policy.name, "bfloat16")
        self.assertEqual(model.backbone.dtype_policy.name, "bfloat16")

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on a task directly (it is ambiguous).
            Task.from_preset("bert_tiny_en_uncased", load_weights=False)
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertTextClassifier.from_preset("gpt2_base_en", load_weights=False)
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            CausalLM.from_preset("hf://spacy/en_core_web_sm")

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

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        task = TextClassifier.from_preset("bert_tiny_en_uncased", num_classes=2)
        task.save_to_preset(save_dir)

        # Check existence of files.
        path = pathlib.Path(save_dir)
        self.assertTrue(os.path.exists(path / CONFIG_FILE))
        self.assertTrue(os.path.exists(path / MODEL_WEIGHTS_FILE))
        self.assertTrue(os.path.exists(path / METADATA_FILE))
        self.assertTrue(os.path.exists(path / TASK_CONFIG_FILE))
        self.assertTrue(os.path.exists(path / TASK_WEIGHTS_FILE))

        # Check the task config (`task.json`).
        task_config = load_json(save_dir, TASK_CONFIG_FILE)
        self.assertTrue("build_config" not in task_config)
        self.assertTrue("compile_config" not in task_config)
        self.assertTrue("backbone" in task_config["config"])
        self.assertTrue("preprocessor" in task_config["config"])

        # Check the preset directory task class.
        self.assertEqual(BertTextClassifier, check_config_class(task_config))

        # Try loading the model from preset directory.
        restored_task = TextClassifier.from_preset(
            save_dir, load_task_extras=True
        )

        # Check the model output.
        data = ["the quick brown fox.", "the slow brown fox."]
        ref_out = task.predict(data)
        new_out = restored_task.predict(data)
        self.assertAllClose(ref_out, new_out)

        # Load without head weights.
        restored_task = TextClassifier.from_preset(
            save_dir, load_task_extras=False, num_classes=2
        )
        data = ["the quick brown fox.", "the slow brown fox."]
        # Full output unequal.
        ref_out = task.predict(data)
        new_out = restored_task.predict(data)
        self.assertNotAllClose(ref_out, new_out)
        # Backbone output equal.
        data = task.preprocessor(data)
        ref_out = task.backbone.predict(data)
        new_out = restored_task.backbone.predict(data)
        self.assertAllClose(ref_out, new_out)

    @pytest.mark.large
    def test_none_preprocessor(self):
        model = TextClassifier.from_preset(
            "bert_tiny_en_uncased",
            preprocessor=None,
            num_classes=2,
        )
        self.assertEqual(model.preprocessor, None)
