# Copyright 2022 The KerasNLP Authors
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
"""Tests for loading pretrained model presets."""

import pytest
import tensorflow as tf

from keras_nlp.models.gpt2.gpt2_models import GPT2
from keras_nlp.models.gpt2.gpt2_preprocessing import GPT2Tokenizer


@pytest.mark.large
class GPT2PresetSmokeTest(tf.test.TestCase):
    """
    A smoke test for GPT-2 presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/gpt2/gpt2_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = GPT2Tokenizer.from_preset(
            "gpt2_base",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [464, 2068, 7586, 21831]
        self.assertAllEqual(outputs, expected_outputs)

    def test_backbone_output(self):
        input_data = {
            "token_ids": tf.constant([[1169, 2068, 7586, 21831, 13]]),
            "padding_mask": tf.constant([[1, 1, 1, 1, 1]]),
        }
        model = GPT2.from_preset("gpt2_base")
        outputs = model(input_data)[0, 0, :5]
        # Outputs from our preset checkpoints should be stable!
        # We should only update these numbers if we are updating a weights file,
        # or have found a bug where output did not match the upstream source.
        expected_outputs = [-0.1116, -0.0375, -0.2624, 0.00891, -0.0061]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)


@pytest.mark.extra_large
class GPT2PresetTest(tf.test.TestCase):
    """
    Test the full enumeration of our preset.

    This only tests all enumeration of our presets and is only run manually.
    Run with:
    `pytest keras_nlp/models/gpt2_presets_test.py --run_extra_large`
    """

    def test_load_gpt2(self):
        for preset in GPT2.presets:
            model = GPT2.from_preset(preset, load_weights=True)
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 1024),
                    dtype=tf.int64,
                    maxval=model.vocabulary_size,
                ),
                "padding_mask": tf.constant([1] * 1024, shape=(1, 1024)),
            }
            model(input_data)

    def test_load_tokenizers(self):
        for preset in GPT2Tokenizer.presets:
            tokenizer = GPT2Tokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")
