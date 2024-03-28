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

import json
import os

import pytest
from absl.testing import parameterized

from keras_nlp import upload_preset
from keras_nlp.models import AlbertClassifier
from keras_nlp.models import BertBackbone
from keras_nlp.models import BertClassifier
from keras_nlp.models import BertTokenizer
from keras_nlp.models import RobertaClassifier
from keras_nlp.tests.test_case import TestCase
from keras_nlp.utils.preset_utils import CONFIG_FILE
from keras_nlp.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.utils.preset_utils import check_config_class
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.preset_utils import save_to_preset


class PresetUtilsTest(TestCase):
    @parameterized.parameters(
        (AlbertClassifier, "albert_base_en_uncased", "sentencepiece"),
        (RobertaClassifier, "roberta_base_en", "bytepair"),
        (BertClassifier, "bert_tiny_en_uncased", "wordpiece"),
    )
    @pytest.mark.keras_3_only
    @pytest.mark.large
    def test_preset_saving(self, cls, preset_name, tokenizer_type):
        save_dir = self.get_temp_dir()
        model = cls.from_preset(preset_name, num_classes=2)
        save_to_preset(model, save_dir)

        if tokenizer_type == "bytepair":
            vocab_filename = "assets/tokenizer/vocabulary.json"
            expected_assets = [
                "assets/tokenizer/vocabulary.json",
                "assets/tokenizer/merges.txt",
            ]
        elif tokenizer_type == "sentencepiece":
            vocab_filename = "assets/tokenizer/vocabulary.spm"
            expected_assets = ["assets/tokenizer/vocabulary.spm"]
        else:
            vocab_filename = "assets/tokenizer/vocabulary.txt"
            expected_assets = ["assets/tokenizer/vocabulary.txt"]

        # Check existence of files
        self.assertTrue(os.path.exists(os.path.join(save_dir, vocab_filename)))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, "model.weights.h5"))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "metadata.json")))

        # Check the model config (`config.json`)
        config_json = open(os.path.join(save_dir, "config.json"), "r").read()
        self.assertTrue(
            "build_config" not in config_json
        )  # Test on raw json to include nested keys
        self.assertTrue(
            "compile_config" not in config_json
        )  # Test on raw json to include nested keys
        config = json.loads(config_json)
        self.assertEqual(set(config["assets"]), set(expected_assets))
        self.assertEqual(config["weights"], "model.weights.h5")

        # Try loading the model from preset directory
        self.assertEqual(cls, check_config_class(save_dir))

        # Try loading the model from preset directory
        restored_model = load_from_preset(save_dir)

        train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        model_input_data = model.preprocessor(*train_data)
        restored_model_input_data = restored_model.preprocessor(*train_data)

        # Check that saved vocab is equal to the original preset vocab
        self.assertAllClose(model_input_data, restored_model_input_data)

        # Check model outputs
        self.assertAllEqual(
            model(model_input_data), restored_model(restored_model_input_data)
        )

    def test_preset_errors(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            AlbertClassifier.from_preset(AlbertClassifier)

        with self.assertRaisesRegex(ValueError, "Unknown preset identifier"):
            AlbertClassifier.from_preset("snaggle://bort/bort/bort")

    def test_upload_empty_preset(self):
        temp_dir = self.get_temp_dir()
        empty_preset = os.path.join(temp_dir, "empty")
        os.mkdir(empty_preset)
        uri = "kaggle://test/test/test"

        with self.assertRaises(FileNotFoundError):
            upload_preset(uri, empty_preset)

    @parameterized.parameters(
        (TOKENIZER_CONFIG_FILE), (CONFIG_FILE), ("model.weights.h5")
    )
    @pytest.mark.keras_3_only
    @pytest.mark.large
    def test_upload_with_missing_file(self, missing_file):
        # Load a model from Kaggle to use as a test model.
        preset = "bert_tiny_en_uncased"
        backbone = BertBackbone.from_preset(preset)
        tokenizer = BertTokenizer.from_preset(preset)

        # Save the model on a local directory.
        temp_dir = self.get_temp_dir()
        local_preset_dir = os.path.join(temp_dir, "bert_preset")
        backbone.save_to_preset(local_preset_dir)
        tokenizer.save_to_preset(local_preset_dir)

        # Delete the file that is supposed to be missing.
        missing_path = os.path.join(local_preset_dir, missing_file)
        os.remove(missing_path)

        # Verify error handling.
        with self.assertRaisesRegex(FileNotFoundError, "is missing"):
            upload_preset("kaggle://test/test/test", local_preset_dir)

    @parameterized.parameters((TOKENIZER_CONFIG_FILE), (CONFIG_FILE))
    @pytest.mark.keras_3_only
    @pytest.mark.large
    def test_upload_with_invalid_json(self, json_file):
        # Load a model from Kaggle to use as a test model.
        preset = "bert_tiny_en_uncased"
        backbone = BertBackbone.from_preset(preset)
        tokenizer = BertTokenizer.from_preset(preset)

        # Save the model on a local directory.
        temp_dir = self.get_temp_dir()
        local_preset_dir = os.path.join(temp_dir, "bert_preset")
        backbone.save_to_preset(local_preset_dir)
        tokenizer.save_to_preset(local_preset_dir)

        # Re-write json file content to an invalid format.
        json_path = os.path.join(local_preset_dir, json_file)
        with open(json_path, "w") as file:
            file.write("Invalid!")

        # Verify error handling.
        with self.assertRaisesRegex(ValueError, "is an invalid json"):
            upload_preset("kaggle://test/test/test", local_preset_dir)
