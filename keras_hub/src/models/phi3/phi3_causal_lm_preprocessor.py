# Copyright 2024 The KerasHub Authors
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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_hub.src.models.phi3.phi3_tokenizer import Phi3Tokenizer


@keras_hub_export("keras_hub.models.Phi3CausalLMPreprocessor")
class Phi3CausalLMPreprocessor(CausalLMPreprocessor):
    """Phi3 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.Phi3CausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.Phi3CausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.Phi3Tokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Default is `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Default is `False`.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Phi3CausalLMPreprocessor.from_preset(
        "phi3_mini_4k_instruct_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        [
            "Avatar 2 is amazing!",
            "Well, I am not sure.",
        ]
    )
    labels = tf.constant([1, 0])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabled sentences.
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = Phi3Backbone
    tokenizer_cls = Phi3Tokenizer
