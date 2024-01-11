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

import numpy as np
import pytest

from keras_nlp.models.gemma.gemma_backbone import GemmaBackbone
from keras_nlp.tests.test_case import TestCase


@pytest.mark.keras_3_only
class GemmaLoraTest(TestCase):
    def setUp(self):
        self._init_kwargs = {
            "vocabulary_size": 50,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 16,
            "head_dim": 16,
            "layer_norm_epsilon": 1e-6,
        }

    def test_lora_fine_tuning(self):
        # Set up backbone and preprocessor.
        backbone = GemmaBackbone(**self._init_kwargs)
        backbone.enable_lora(4)
        # 4 layers, 2 weights per layer
        self.assertLen(backbone.trainable_weights, 4 * 2)
        self.assertLen(backbone.non_trainable_weights, 20)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))

        # Test fine-tuning
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(temp_filepath)
        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

    def test_lora_saving_and_reloading(self):
        backbone = GemmaBackbone(**self._init_kwargs)
        initial_model_filepath = os.path.join(
            self.get_temp_dir(), "base.weights.h5"
        )
        backbone.save_weights(initial_model_filepath)

        backbone.enable_lora(4)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        lora_filepath = os.path.join(self.get_temp_dir(), "lora_model.lora.h5")
        backbone.save_lora_weights(lora_filepath)

        # New backbone with same initial weights
        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(initial_model_filepath)
        new_backbone.enable_lora(4)
        new_backbone.load_lora_weights(lora_filepath)

        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

        # Test exceptions
        backbone = GemmaBackbone(**self._init_kwargs)
        with self.assertRaisesRegex(ValueError, "no lora-enabled layers"):
            backbone.save_lora_weights(lora_filepath)
        backbone.enable_lora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_lora_weights(lora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_lora_weights("bad_filepath")
