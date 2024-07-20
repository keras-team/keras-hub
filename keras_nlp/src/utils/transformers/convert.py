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


from keras_nlp.src.utils.transformers.convert_bert import load_bert_backbone
from keras_nlp.src.utils.transformers.convert_bert import load_bert_tokenizer
from keras_nlp.src.utils.transformers.convert_gemma import load_gemma_backbone
from keras_nlp.src.utils.transformers.convert_gemma import load_gemma_tokenizer
from keras_nlp.src.utils.transformers.convert_gpt2 import load_gpt2_backbone
from keras_nlp.src.utils.transformers.convert_gpt2 import load_gpt2_tokenizer
from keras_nlp.src.utils.transformers.convert_llama3 import load_llama3_backbone
from keras_nlp.src.utils.transformers.convert_llama3 import (
    load_llama3_tokenizer,
)
from keras_nlp.src.utils.transformers.convert_pali_gemma import (
    load_pali_gemma_backbone,
)
from keras_nlp.src.utils.transformers.convert_pali_gemma import (
    load_pali_gemma_tokenizer,
)


def load_transformers_backbone(cls, preset, load_weights):
    """
    Load a Transformer model config and weights as a KerasNLP backbone.

    Args:
        cls (class): Keras model class.
        preset (str): Preset configuration name.
        load_weights (bool): Whether to load the weights.

    Returns:
        backbone: Initialized Keras model backbone.
    """
    if cls is None:
        raise ValueError("Backbone class is None")
    if cls.__name__ == "BertBackbone":
        return load_bert_backbone(cls, preset, load_weights)
    if cls.__name__ == "GemmaBackbone":
        return load_gemma_backbone(cls, preset, load_weights)
    if cls.__name__ == "Llama3Backbone":
        return load_llama3_backbone(cls, preset, load_weights)
    if cls.__name__ == "PaliGemmaBackbone":
        return load_pali_gemma_backbone(cls, preset, load_weights)
    if cls.__name__ == "GPT2Backbone":
        return load_gpt2_backbone(cls, preset, load_weights)
    raise ValueError(
        f"{cls} has not been ported from the Hugging Face format yet. "
        "Please check Hugging Face Hub for the Keras model. "
    )


def load_transformers_tokenizer(cls, preset):
    """
    Load a Transformer tokenizer assets as a KerasNLP tokenizer.

    Args:
        cls (class): Tokenizer class.
        preset (str): Preset configuration name.

    Returns:
        tokenizer: Initialized tokenizer.
    """
    if cls is None:
        raise ValueError("Tokenizer class is None")
    if cls.__name__ == "BertTokenizer":
        return load_bert_tokenizer(cls, preset)
    if cls.__name__ == "GemmaTokenizer":
        return load_gemma_tokenizer(cls, preset)
    if cls.__name__ == "Llama3Tokenizer":
        return load_llama3_tokenizer(cls, preset)
    if cls.__name__ == "PaliGemmaTokenizer":
        return load_pali_gemma_tokenizer(cls, preset)
    if cls.__name__ == "GPT2Tokenizer":
        return load_gpt2_tokenizer(cls, preset)
    raise ValueError(
        f"{cls} has not been ported from the Hugging Face format yet. "
        "Please check Hugging Face Hub for the Keras model. "
    )
