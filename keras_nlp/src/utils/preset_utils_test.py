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

import json
import os

import keras
import pytest
from absl.testing import parameterized

from keras_nlp.src.models.albert.albert_text_classifier import (
    AlbertTextClassifier,
)
from keras_nlp.src.models.bert.bert_backbone import BertBackbone
from keras_nlp.src.models.bert.bert_tokenizer import BertTokenizer
from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.utils.keras_utils import has_quantization_support
from keras_nlp.src.utils.preset_utils import CONFIG_FILE
from keras_nlp.src.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import load_serialized_object
from keras_nlp.src.utils.preset_utils import upload_preset


class PresetUtilsTest(TestCase):
    def test_preset_errors(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            AlbertTextClassifier.from_preset(AlbertTextClassifier)

        with self.assertRaisesRegex(ValueError, "Unknown preset identifier"):
            AlbertTextClassifier.from_preset("snaggle://bort/bort/bort")

        backbone = BertBackbone.from_preset("bert_tiny_en_uncased")
        preset_dir = self.get_temp_dir()
        config = keras.utils.serialize_keras_object(backbone)
        config["registered_name"] = "keras_nlp>BortBackbone"
        with open(os.path.join(preset_dir, CONFIG_FILE), "w") as config_file:
            config_file.write(json.dumps(config, indent=4))
        with self.assertRaisesRegex(ValueError, "class keras_nlp>BortBackbone"):
            BertBackbone.from_preset(preset_dir)

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

    @parameterized.named_parameters(
        ("gemma2_2b_en", "gemma2_2b_en", "bfloat16", False),
        ("llama2_7b_en_int8", "llama2_7b_en_int8", "bfloat16", True),
    )
    @pytest.mark.extra_large
    def test_load_serialized_object(self, preset, dtype, is_quantized):
        if is_quantized and not has_quantization_support():
            self.skipTest("This version of Keras doesn't support quantization.")

        model = load_serialized_object(preset, dtype=dtype)
        if is_quantized:
            self.assertEqual(model.dtype_policy.name, "map_bfloat16")
        else:
            self.assertEqual(model.dtype_policy.name, "bfloat16")
