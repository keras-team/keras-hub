# Copyright 2022 The KerasNLP Authors
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

"""RoBERTa masked language model preprocessor layer."""

from absl import logging
from tensorflow import keras

from keras_nlp.layers.masked_lm_mask_generator import MaskedLMMaskGenerator
from keras_nlp.models.roberta.roberta_preprocessor import RobertaPreprocessor
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras.utils.register_keras_serializable(package="keras_nlp")
class RobertaMaskedLMPreprocessor(RobertaPreprocessor):
    """RoBERTa preprocessing for the masked language modeling task.

    This preprocessing layer will prepare inputs for a masked language modeling
    task. It is primarily intended for use with the
    `keras_nlp.models.RobertaMaskedLM` task model. Preprocessing will occur in
    multiple steps.

    - Tokenize any number of input segments using the `tokenizer`.
    - Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
      `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
      entire sequence, `"</s></s>"` between each segment,
      and a `"</s>"` at the end of the entire sequence.
    - Randomly select non-special tokens to mask, controlled by
      `mask_selection_rate`.
    - Construct a `(x, y, sample_weight)` tuple suitable for training with a
      `keras_nlp.models.RobertaMaskedLM` task model.

    Args:
        tokenizer: A `keras_nlp.models.RobertaTokenizer` instance.
        sequence_length: The length of the packed inputs.
        mask_selection_rate: The probability an input token will be dynamically
            masked.
        mask_selection_length: The maximum number of masked tokens supported
            by the layer.
        mask_token_rate: float, defaults to 0.8. `mask_token_rate` must be
            between 0 and 1 which indicates how often the mask_token is
            substituted for tokens selected for masking.
        random_token_rate: float, defaults to 0.1. `random_token_rate` must be
            between 0 and 1 which indicates how often a random token is
            substituted for tokens selected for masking. Default is 0.1.
            Note: mask_token_rate + random_token_rate <= 1,  and for
            (1 - mask_token_rate - random_token_rate), the token will not be
            changed.
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
    preprocessor = keras_nlp.models.RobertaMaskedLMPreprocessor.from_preset(
        "roberta_base_en"
    )

    # Tokenize and mask a single sentence.
    sentence = tf.constant("The quick brown fox jumped.")
    preprocessor(sentence)

    # Tokenize and mask a batch of sentences.
    sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    preprocessor(sentences)

    # Tokenize and mask a dataset of sentences.
    features = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    ds = tf.data.Dataset.from_tensor_slices((features))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Alternatively, you can create a preprocessor from your own vocabulary.
    # The usage is exactly the same as above.
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick", "Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_nlp.models.RobertaTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.RobertaMaskedLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=8,
    )
    preprocessor("a quick fox")
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        mask_selection_rate=0.15,
        mask_selection_length=96,
        mask_token_rate=0.8,
        random_token_rate=0.1,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            sequence_length=sequence_length,
            truncate=truncate,
            **kwargs,
        )

        self.masker = MaskedLMMaskGenerator(
            mask_selection_rate=mask_selection_rate,
            mask_selection_length=mask_selection_length,
            mask_token_rate=mask_token_rate,
            random_token_rate=random_token_rate,
            vocabulary_size=tokenizer.vocabulary_size(),
            mask_token_id=tokenizer.mask_token_id,
            unselectable_token_ids=[
                tokenizer.start_token_id,
                tokenizer.end_token_id,
                tokenizer.pad_token_id,
            ],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mask_selection_rate": self.masker.mask_selection_rate,
                "mask_selection_length": self.masker.mask_selection_length,
                "mask_token_rate": self.masker.mask_token_rate,
                "random_token_rate": self.masker.random_token_rate,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                f"{self.__class__.__name__} generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored."
            )

        x = super().call(x)
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        masker_outputs = self.masker(token_ids)
        x = {
            "token_ids": masker_outputs["token_ids"],
            "padding_mask": padding_mask,
            "mask_positions": masker_outputs["mask_positions"],
        }
        y = masker_outputs["mask_ids"]
        sample_weight = masker_outputs["mask_weights"]
        return pack_x_y_sample_weight(x, y, sample_weight)
