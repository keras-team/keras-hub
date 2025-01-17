import os
import pathlib

import keras
import numpy as np
import pytest

from keras_hub.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.task import Task
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import METADATA_FILE
from keras_hub.src.utils.preset_utils import MODEL_WEIGHTS_FILE
from keras_hub.src.utils.preset_utils import TASK_CONFIG_FILE
from keras_hub.src.utils.preset_utils import TASK_WEIGHTS_FILE
from keras_hub.src.utils.preset_utils import check_config_class
from keras_hub.src.utils.preset_utils import load_json


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
        restored_task = TextClassifier.from_preset(save_dir)

        # Check the model output.
        data = ["the quick brown fox.", "the slow brown fox."]
        ref_out = task.predict(data)
        new_out = restored_task.predict(data)
        self.assertAllClose(ref_out, new_out)

        # Load classifier head with random weights.
        restored_task = TextClassifier.from_preset(save_dir, num_classes=2)
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

        # Check setting dtype.
        restored_task = TextClassifier.from_preset(save_dir, dtype="float16")
        self.assertEqual("float16", restored_task.backbone.dtype_policy.name)

    @pytest.mark.large
    def test_save_to_preset_custom_backbone_and_preprocessor(self):
        preprocessor = keras.layers.Rescaling(1 / 255.0)
        inputs = keras.Input(shape=(None, None, 3))
        outputs = keras.layers.Dense(8)(inputs)
        backbone = keras.Model(inputs, outputs)
        task = ImageClassifier(
            backbone=backbone,
            preprocessor=preprocessor,
            num_classes=10,
        )

        save_dir = self.get_temp_dir()
        task.save_to_preset(save_dir)
        batch = np.random.randint(0, 256, size=(2, 224, 224, 3))
        expected = task.predict(batch)

        restored_task = ImageClassifier.from_preset(save_dir)
        actual = restored_task.predict(batch)
        self.assertAllClose(expected, actual)
