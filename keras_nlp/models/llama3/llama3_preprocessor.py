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
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_nlp.models.llama.llama_preprocessor import LlamaPreprocessor


@keras_nlp_export("keras_nlp.models.Llama3Preprocessor")
class Llama3Preprocessor(LlamaPreprocessor):
    tokenizer_cls = Llama3Tokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=False,
        **kwargs
    ):
        super().__init__(
            tokenizer, sequence_length, add_start_token, add_end_token, **kwargs
        )
