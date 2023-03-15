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
"""BART preprocessor layer."""

import copy

from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty

PRESET_NAMES = ", ".join(list(backbone_presets))


@keras_nlp_export("keras_nlp.models.BartPreprocessor")
class BartPreprocessor(Preprocessor):
    """A BART preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     - Tokenize any number of input segments using the `tokenizer`.
     - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
     - Construct a dictionary with keys `"token_ids"`, `"segment_ids"`,
       `"padding_mask"`, that can be passed directly to a BERT model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    The call method of this layer accepts three arguments, `x`, `y`, and
    `sample_weight`. `x` can be a python string or tensor representing a single
    segment, a list of python strings representing a batch of single segments,
    or a list of tensors representing multiple segments to be packed together.
    `y` and `sample_weight` are both optional, can have any format, and will be
    passed through unaltered.

    Special care should be taken when using `tf.data` to map over an unlabeled
    tuple of string segments. `tf.data.Dataset.map` will unpack this tuple
    directly into the call arguments of this layer, rather than forward all
    argument to `x`. To handle this case, it is recommended to  explicitly call
    the layer, e.g. `ds.map(lambda seg1, seg2: preprocessor(x=(seg1, seg2)))`.

    Args:
        tokenizer: A `keras_nlp.models.BartTokenizer` instance.
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

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bert_base_en_uncased")

    # Tokenize and pack a single sentence.
    sentence = tf.constant("The quick brown fox jumped.")
    preprocessor(sentence)
    # Same output.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and a batch of single sentences.
    sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    preprocessor(sentences)
    # Same output.
    preprocessor(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )

    # Tokenize and pack a sentence pair.
    first_sentence = tf.constant("The quick brown fox jumped.")
    second_sentence = tf.constant("The fox tripped.")
    preprocessor((first_sentence, second_sentence))

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    labels = tf.constant([0, 1])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess sentence pairs.
    first_sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    second_sentences = tf.constant(
        ["The fox tripped.", "Oh look, a whale."]
    )
    labels = tf.constant([1, 1])
    ds = tf.data.Dataset.from_tensor_slices(
        (
            (first_sentences, second_sentences), labels
        )
    )
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabeled sentence pairs.
    first_sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    second_sentences = tf.constant(
        ["The fox tripped.", "Oh look, a whale."]
    )
    ds = tf.data.Dataset.from_tensor_slices((first_sentences, second_sentences))
    # Watch out for tf.data's default unpacking of tuples here!
    # Best to invoke the `preprocessor` directly in this case.
    ds = ds.map(
        lambda s1, s2: preprocessor(x=(s1, s2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Alternatively, you can create a preprocessor from your own vocabulary.
    # The usage is exactly the same as shown above.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab += ["The", "qu", "##ick", "br", "##own", "fox", "tripped"]
    vocab += ["Call", "me", "Ish", "##mael", "."]
    vocab += ["Oh", "look", "a", "whale"]
    vocab += ["I", "forgot", "my", "home", "##work"]
    tokenizer = keras_nlp.models.BartTokenizer(vocabulary=vocab)
    preprocessor = keras_nlp.models.BartPreprocessor(tokenizer)
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError(
                "The input to `BartPreprocessor` must be a tuple "
                "`(encoder_inputs, decoder_inputs)`."
            )

        if y is not None:
            logging.warning(
                "You are explicitly passing `y` to `BartPreprocessor`. However, "
                "`y` is inferred from decoder inputs, and will be ignored."
            )

        encoder_inputs, decoder_inputs = x

        encoder_inputs = convert_inputs_to_list_of_tensor_segments(
            encoder_inputs
        )
        encoder_inputs = [self.tokenizer(segment) for segment in encoder_inputs]
        encoder_token_ids, _ = self.packer(encoder_inputs)

        decoder_inputs = convert_inputs_to_list_of_tensor_segments(
            decoder_inputs
        )
        decoder_inputs = [self.tokenizer(segment) for segment in decoder_inputs]
        decoder_token_ids, _ = self.packer(decoder_inputs)

        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_token_ids
            != self.tokenizer.pad_token_id,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_token_ids
            != self.tokenizer.pad_token_id,
        }

        # Get the labels by shifting the decoder inputs one place to the left.
        if decoder_token_ids.shape.rank == 1:
            y = decoder_token_ids[1:]
        else:
            y = decoder_token_ids[:, 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def tokenizer_cls(cls):
        return BartTokenizer

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})
