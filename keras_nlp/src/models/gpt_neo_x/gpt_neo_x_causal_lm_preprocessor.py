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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_nlp.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_nlp.src.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer


@keras_nlp_export("keras_nlp.models.GPTNeoXCausalLMPreprocessor")
class GPTNeoXCausalLMPreprocessor(CausalLMPreprocessor):
    """GPT-NeoX Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_nlp.models.GPTNeoXCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_nlp.models.GPTNeoXCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_nlp.models.GPTNeoXTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    """

    backbone_cls = GPTNeoXBackbone
    tokenizer_cls = GPTNeoXTokenizer
