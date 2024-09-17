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

import numpy as np
import pytest

from keras_nlp.src.layers.preprocessing.image_converter import ImageConverter
from keras_nlp.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_image_converter import (
    PaliGemmaImageConverter,
)
from keras_nlp.src.tests.test_case import TestCase


class ImageConverterTest(TestCase):
    def test_preset_accessors(self):
        pali_gemma_presets = set(PaliGemmaImageConverter.presets.keys())
        all_presets = set(ImageConverter.presets.keys())
        self.assertContainsSubset(pali_gemma_presets, all_presets)
        self.assertIn("pali_gemma_3b_mix_224", pali_gemma_presets)
        self.assertIn("pali_gemma_3b_mix_224", all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            ImageConverter.from_preset("pali_gemma_3b_mix_224"),
            PaliGemmaImageConverter,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            ImageConverter.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            ImageConverter.from_preset("hf://spacy/en_core_web_sm")

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        converter = ImageConverter.from_preset(
            "pali_gemma_3b_mix_224",
            interpolation="nearest",
        )
        converter.save_to_preset(save_dir)
        # Save a tiny backbone so the preset is valid.
        backbone = PaliGemmaBackbone(
            vocabulary_size=100,
            image_size=224,
            num_layers=1,
            num_query_heads=1,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=8,
            vit_patch_size=14,
            vit_num_heads=1,
            vit_hidden_dim=8,
            vit_num_layers=1,
        )
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        path = pathlib.Path(save_dir)
        self.assertTrue(os.path.exists(path / "image_converter.json"))

        # Check loading.
        restored = ImageConverter.from_preset(save_dir)
        test_image = np.random.rand(100, 100, 3) * 255
        self.assertAllClose(restored(test_image), converter(test_image))
