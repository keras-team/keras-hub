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

import io

import pytest
import sentencepiece

from keras_nlp.models.f_net.f_net_backbone import FNetBackbone
from keras_nlp.models.f_net.f_net_masked_lm import FNetMaskedLM
from keras_nlp.models.f_net.f_net_masked_lm_preprocessor import (
    FNetMaskedLMPreprocessor,
)
from keras_nlp.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_nlp.tests.test_case import TestCase


class FNetMaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        vocab_data = ["the quick brown fox", "the earth is round"]
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(vocab_data),
            model_writer=bytes_io,
            vocab_size=12,
            model_type="WORD",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            user_defined_symbols="[MASK]",
        )
        self.preprocessor = FNetMaskedLMPreprocessor(
            FNetTokenizer(proto=bytes_io.getvalue()),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=5,
        )
        self.backbone = FNetBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_masked_lm_basics(self):
        self.run_task_test(
            cls=FNetMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 12),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=FNetMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FNetMaskedLM.presets:
            self.run_preset_test(
                cls=FNetMaskedLM,
                preset=preset,
                input_data=self.input_data,
            )
