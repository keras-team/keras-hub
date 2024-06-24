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
"""Convert huggingface models to KerasNLP."""


from keras_nlp.src.utils.transformers.convert_gemma import load_gemma_backbone
from keras_nlp.src.utils.transformers.convert_gemma import load_gemma_tokenizer
from keras_nlp.src.utils.transformers.convert_llama3 import load_llama3_backbone
from keras_nlp.src.utils.transformers.convert_llama3 import (
    load_llama3_tokenizer,
)


def load_transformers_backbone(cls, preset, load_weights):
    if cls is None:
        raise ValueError("Backbone class is None")
    if cls.__name__ == "GemmaBackbone":
        return load_gemma_backbone(cls, preset, load_weights)
    if cls.__name__ == "Llama3Backbone":
        return load_llama3_backbone(cls, preset, load_weights)
    raise ValueError(
        f"{cls} has not been ported from the Hugging Face format yet. "
        "Please check Hugging Face Hub for the Keras model. "
    )


def load_transformers_tokenizer(cls, preset):
    if cls is None:
        raise ValueError("Tokenizer class is None")
    if cls.__name__ == "GemmaTokenizer":
        return load_gemma_tokenizer(cls, preset)
    if cls.__name__ == "Llama3Tokenizer":
        return load_llama3_tokenizer(cls, preset)
    raise ValueError(
        f"{cls} has not been ported from the Hugging Face format yet. "
        "Please check Hugging Face Hub for the Keras model. "
    )
