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
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer


@keras_hub_export("keras_hub.models.GemmaCausalLMPreprocessor")
class GemmaCausalLMPreprocessor(CausalLMPreprocessor):
    """Gemma Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.GemmaCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.GemmaCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.GemmaTokenizer` instance.
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

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
        "gemma_2b_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Apply tokenization to a `tf.data.Dataset`.
    features = tf.constant(["The quick brown fox.", "Call me Ishmael."])
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare tokens for generation (no end token).
    preprocessor.generate_preprocess(["The quick brown fox jumped."])

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        'token_ids': np.array([[2, 714, 4320, 8426, 25341, 32292, 235265, 0]]),
        'padding_mask': np.array([[ 1,  1,  1,  1,  1,  1,  1, 0]]),
    })
    ```
    """

    backbone_cls = GemmaBackbone
    tokenizer_cls = GemmaTokenizer
