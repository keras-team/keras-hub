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

from keras_nlp.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_nlp.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = GemmaCausalLM.from_preset("hf://ariG23498/tiny-gemma-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    # TODO: compare numerics with huggingface model
