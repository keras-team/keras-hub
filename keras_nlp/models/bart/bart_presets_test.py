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
"""Tests for loading pretrained model presets."""

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer


@pytest.mark.large
class BartPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for BART presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/bart/bart_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = BartTokenizer.from_preset(
            "bart_base_en",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [133, 2119, 6219, 23602, 4]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "encoder_token_ids": tf.constant([[0, 133, 2119, 2]]),
            "encoder_padding_mask": tf.constant([[1, 1, 1, 1]]),
            "decoder_token_ids": tf.constant([[0, 7199, 14, 2119, 2]]),
            "decoder_padding_mask": tf.constant([[1, 1, 1, 1, 1]]),
        }
        model = BartBackbone.from_preset(
            "bart_base_en", load_weights=load_weights
        )
        outputs = model(input_data)
        if load_weights:
            encoder_output = outputs["encoder_sequence_output"][0, 0, :5]
            expected_encoder_output = [-0.033, 0.013, -0.003, -0.012, -0.002]
            decoder_output = outputs["decoder_sequence_output"][0, 0, :5]
            expected_decoder_output = [2.516, 2.489, 0.695, 8.057, 1.245]

            self.assertAllClose(
                encoder_output, expected_encoder_output, atol=0.01, rtol=0.01
            )
            self.assertAllClose(
                decoder_output, expected_decoder_output, atol=0.01, rtol=0.01
            )

    @parameterized.named_parameters(
        ("bart_tokenizer", BartTokenizer),
        ("bart", BartBackbone),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("bart_tokenizer", BartTokenizer),
        ("bart", BartBackbone),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("bart_base_en_clowntown")


@pytest.mark.extra_large
class BartPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This tests every BART preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/bart/bart_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_bart(self, load_weights):
        for preset in BartBackbone.presets:
            model = BartBackbone.from_preset(preset, load_weights=load_weights)
            input_data = {
                "encoder_token_ids": tf.random.uniform(
                    shape=(1, 1024),
                    dtype=tf.int64,
                    maxval=model.vocabulary_size,
                ),
                "encoder_padding_mask": tf.constant(
                    [1] * 768 + [0] * 256, shape=(1, 1024)
                ),
                "decoder_token_ids": tf.random.uniform(
                    shape=(1, 1024),
                    dtype=tf.int64,
                    maxval=model.vocabulary_size,
                ),
                "decoder_padding_mask": tf.constant(
                    [1] * 489 + [0] * 535, shape=(1, 1024)
                ),
            }
            model(input_data)

    def test_load_tokenizers(self):
        for preset in BartTokenizer.presets:
            tokenizer = BartTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")
