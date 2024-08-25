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
import pytest

from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.bart.bart_backbone import BartBackbone
from keras_nlp.src.models.bart.bart_seq_2_seq_lm import BartSeq2SeqLM
from keras_nlp.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_nlp.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = BartSeq2SeqLM.from_preset("hf://cosmo3769/tiny-bart-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = Seq2SeqLM.from_preset(
            "hf://cosmo3769/tiny-bart-test",
            load_weights=False,
        )
        self.assertIsInstance(model, BartSeq2SeqLM)
        model = Backbone.from_preset(
            "hf://cosmo3769/tiny-bart-test",
            load_weights=False,
        )
        self.assertIsInstance(model, BartBackbone)

    # TODO: compare numerics with huggingface model
