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
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_nlp.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_nlp.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = Llama3CausalLM.from_preset("hf://ariG23498/tiny-llama3-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://ariG23498/tiny-llama3-test",
            load_weights=False,
        )
        self.assertIsInstance(model, Llama3CausalLM)
        model = Backbone.from_preset(
            "hf://ariG23498/tiny-llama3-test",
            load_weights=False,
        )
        self.assertIsInstance(model, Llama3Backbone)

    # TODO: compare numerics with huggingface model
