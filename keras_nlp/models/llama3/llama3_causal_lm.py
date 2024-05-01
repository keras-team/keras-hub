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
from keras_nlp.models.llama3.llama3_backbone import Llama3Backbone
from keras_nlp.models.llama3.llama3_causal_lm_preprocessor import (
    Llama3CausalLMPreprocessor,
)
from keras_nlp.models.llama.llama_causal_lm import LlamaCausalLM


class Llama3CausalLM(LlamaCausalLM):
    backbone_cls = Llama3Backbone
    preprocessor_cls = Llama3CausalLMPreprocessor
