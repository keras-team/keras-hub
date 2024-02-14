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
import tensorflow as tf

from keras_nlp.models.bart.bart_preprocessor import BartPreprocessor
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer
from keras_nlp.tests.test_case import TestCase


class BartPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "air", "Ġair", "plane", "Ġat"]
        self.vocab += ["port", "<mask>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = BartTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "encoder_sequence_length": 5,
            "decoder_sequence_length": 8,
        }
        self.input_data = (
            {
                "encoder_text": [" airplane at airport"],
                "decoder_text": [" airplane airport"],
            },
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=BartPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "encoder_token_ids": [[0, 4, 5, 6, 2]],
                    "encoder_padding_mask": [[1, 1, 1, 1, 1]],
                    "decoder_token_ids": [[2, 0, 4, 5, 4, 7, 2, 1]],
                    "decoder_padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
            token_id_key="decoder_token_ids",
        )

    def test_error_multi_segment_input(self):
        preprocessor = BartPreprocessor(**self.init_kwargs)
        input_data = {
            "encoder_text": (
                tf.constant([" airplane at airport"] * 2),
                tf.constant([" airplane"] * 2),
            ),
            "decoder_text": (
                tf.constant([" kohli is the best"] * 2),
                tf.constant([" kohli"] * 2),
            ),
        }
        with self.assertRaises(ValueError):
            preprocessor(input_data)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BartPreprocessor.presets:
            self.run_preset_test(
                cls=BartPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
