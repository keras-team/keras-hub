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

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.f_net.f_net_backbone import FNetBackbone
from keras_nlp.src.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_nlp.src.models.masked_lm_preprocessor import MaskedLMPreprocessor
from keras_nlp.src.utils.tensor_utils import preprocessing_function


@keras_nlp_export("keras_nlp.models.FNetMaskedLMPreprocessor")
class FNetMaskedLMPreprocessor(MaskedLMPreprocessor):
    """FNet preprocessing for the masked language modeling task.

    This preprocessing layer will prepare inputs for a masked language modeling
    task. It is primarily intended for use with the
    `keras_nlp.models.FNetMaskedLM` task model. Preprocessing will occur in
    multiple steps.

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
      `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
      entire sequence, `"</s></s>"` between each segment,
      and a `"</s>"` at the end of the entire sequence.
    3. Randomly select non-special tokens to mask, controlled by
      `mask_selection_rate`.
    4. Construct a `(x, y, sample_weight)` tuple suitable for training with a
      `keras_nlp.models.FNetMaskedLM` task model.

    Args:
        tokenizer: A `keras_nlp.models.FNetTokenizer` instance.
        sequence_length: The length of the packed inputs.
        mask_selection_rate: The probability an input token will be dynamically
            masked.
        mask_selection_length: The maximum number of masked tokens supported
            by the layer.
        mask_token_rate: float. `mask_token_rate` must be
            between 0 and 1 which indicates how often the mask_token is
            substituted for tokens selected for masking. Defaults to `0.8`.
        random_token_rate: float. `random_token_rate` must be
            between 0 and 1 which indicates how often a random token is
            substituted for tokens selected for masking.
            Note: mask_token_rate + random_token_rate <= 1,  and for
            (1 - mask_token_rate - random_token_rate), the token will not be
            changed. Defaults to `0.1`.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It supports an arbitrary number of segments.

    Examples:

    Directly calling the layer on data.
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.FNetMaskedLMPreprocessor.from_preset(
        "f_net_base_en"
    )

    # Tokenize and mask a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and mask a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Tokenize and mask sentence pairs.
    # In this case, always convert input to tensors before calling the layer.
    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])
    preprocessor((first, second))
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.FNetMaskedLMPreprocessor.from_preset(
        "f_net_base_en"
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])

    # Map single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Alternatively, you can create a preprocessor from your own vocabulary.
    vocab_data = tf.data.Dataset.from_tensor_slices(
        ["the quick brown fox", "the earth is round"]
    )

    # Map sentence pairs.
    ds = tf.data.Dataset.from_tensor_slices((first, second))
    # Watch out for tf.data's default unpacking of tuples here!
    # Best to invoke the `preprocessor` directly in this case.
    ds = ds.map(
        lambda first, second: preprocessor(x=(first, second)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    backbone_cls = FNetBackbone
    tokenizer_cls = FNetTokenizer

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        output = super().call(x, y=y, sample_weight=sample_weight)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        # FNet has not padding mask.
        del x["padding_mask"]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
