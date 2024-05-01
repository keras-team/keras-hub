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
import os

import numpy as np
import pytest

from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.bert.bert_backbone import BertBackbone
from keras_nlp.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.utils.preset_utils import CONFIG_FILE
from keras_nlp.src.utils.preset_utils import METADATA_FILE
from keras_nlp.src.utils.preset_utils import MODEL_WEIGHTS_FILE
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import load_config


class TestTask(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertBackbone.presets.keys())
        gpt2_presets = set(GPT2Backbone.presets.keys())
        all_presets = set(Backbone.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Backbone.from_preset("bert_tiny_en_uncased", load_weights=False),
            BertBackbone,
        )
        self.assertIsInstance(
            Backbone.from_preset("gpt2_base_en", load_weights=False),
            GPT2Backbone,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            GPT2Backbone.from_preset("bert_tiny_en_uncased", load_weights=False)
        with self.assertRaisesRegex(
            FileNotFoundError, f"doesn't have a file named `{METADATA_FILE}`"
        ):
            # No loading on a non-keras model.
            Backbone.from_preset("hf://google-bert/bert-base-uncased")

    @pytest.mark.keras_3_only
    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        backbone = Backbone.from_preset("bert_tiny_en_uncased")
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        self.assertTrue(os.path.exists(os.path.join(save_dir, CONFIG_FILE)))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, MODEL_WEIGHTS_FILE))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, METADATA_FILE)))

        # Check the backbone config (`config.json`).
        backbone_config = load_config(save_dir, CONFIG_FILE)
        self.assertTrue("build_config" not in backbone_config)
        self.assertTrue("compile_config" not in backbone_config)

        # Try config class.
        self.assertEqual(BertBackbone, check_config_class(save_dir))

        # Try loading the model from preset directory.
        restored_backbone = Backbone.from_preset(save_dir)

        data = {
            "token_ids": np.ones(shape=(2, 10), dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
            "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
        }

        # Check the model output.
        ref_out = backbone(data)
        new_out = restored_backbone(data)
        self.assertAllClose(ref_out, new_out)
