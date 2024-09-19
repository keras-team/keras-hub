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

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)
from keras_hub.src.models.masked_lm_preprocessor import MaskedLMPreprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.DistilBertMaskedLMPreprocessor")
class DistilBertMaskedLMPreprocessor(MaskedLMPreprocessor):
    """DistilBERT preprocessing for the masked language modeling task.

    This preprocessing layer will prepare inputs for a masked language modeling
    task. It is primarily intended for use with the
    `keras_hub.models.DistilBertMaskedLM` task model. Preprocessing will occur in
    multiple steps.

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
    3. Randomly select non-special tokens to mask, controlled by
      `mask_selection_rate`.
    4. Construct a `(x, y, sample_weight)` tuple suitable for training with a
      `keras_hub.models.DistilBertMaskedLM` task model.

    Args:
        tokenizer: A `keras_hub.models.DistilBertTokenizer` instance.
        sequence_length: int. The length of the packed inputs.
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
        mask_selection_rate: float. The probability an input token will be
            dynamically masked.
        mask_selection_length: int. The maximum number of masked tokens
            in a given sample.
        mask_token_rate: float. The probability the a selected token will be
            replaced with the mask token.
        random_token_rate: float. The probability the a selected token will be
            replaced with a random token from the vocabulary. A selected token
            will be left as is with probability
            `1 - mask_token_rate - random_token_rate`.

    Call arguments:
        x: A tensor of single string sequences, or a tuple of multiple
            tensor sequences to be packed together. Inputs may be batched or
            unbatched. For single sequences, raw python inputs will be converted
            to tensors. For multiple sequences, pass tensors directly.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.DistilBertMaskedLMPreprocessor.from_preset(
        "distil_bert_base_en_uncased"
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
    preprocessor = keras_hub.models.DistilBertMaskedLMPreprocessor.from_preset(
        "distil_bert_base_en_uncased"
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])

    # Map single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

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

    backbone_cls = DistilBertBackbone
    tokenizer_cls = DistilBertTokenizer

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        output = super().call(x, y=y, sample_weight=sample_weight)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        # Backbone has no segment ID input.
        del x["segment_ids"]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
