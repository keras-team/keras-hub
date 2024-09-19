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
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_hub.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export(
    [
        "keras_hub.models.RobertaTextClassifierPreprocessor",
        "keras_hub.models.RobertaPreprocessor",
    ]
)
class RobertaTextClassifierPreprocessor(TextClassifierPreprocessor):
    """A RoBERTa preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
       `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
       entire sequence, `"</s></s>"` at the end of each segment, save the last
       and a `"</s>"` at the end of the entire sequence.
    3. Construct a dictionary with keys `"token_ids"`, `"padding_mask"` that
       can be passed directly to a RoBERTa model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.RobertaTokenizer` instance.
        sequence_length: The length of the packed inputs.
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

    Call arguments:
        x: A tensor of single string sequences, or a tuple of multiple
            tensor sequences to be packed together. Inputs may be batched or
            unbatched. For single sequences, raw python inputs will be converted
            to tensors. For multiple sequences, pass tensors directly.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.


    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
        "roberta_base_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Preprocess a batch of sentence pairs.
    # When handling multiple sequences, always convert to tensors first!
    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])
    preprocessor((first, second))

    # Custom vocabulary.
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick", "Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_hub.models.RobertaTokenizer(
        vocabulary=vocab,
        merges=merges
    )
    preprocessor = keras_hub.models.RobertaTextClassifierPreprocessor(tokenizer)
    preprocessor("a quick fox")
    ```
    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
        "roberta_base_en"
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    second = tf.constant(["The fox tripped.", "Oh look, a whale."])
    label = tf.constant([1, 1])

    # Map labeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices((first, label))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map labeled sentence pairs.
    ds = tf.data.Dataset.from_tensor_slices(((first, second), label))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled sentence pairs.
    ds = tf.data.Dataset.from_tensor_slices((first, second))
    # Watch out for tf.data's default unpacking of tuples here!
    # Best to invoke the `preprocessor` directly in this case.
    ds = ds.map(
        lambda first, second: preprocessor(x=(first, second)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    backbone_cls = RobertaBackbone
    tokenizer_cls = RobertaTokenizer

    def build(self, input_shape):
        # Roberta is doubles up the sep token, so we override build.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            sep_value=[self.tokenizer.end_token_id] * 2,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.built = True

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        output = super().call(x, y=y, sample_weight=sample_weight)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        # Backbone has no segment ID input.
        del x["segment_ids"]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
