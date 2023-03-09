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

"""DistilBERT masked language model preprocessor layer."""

from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.masked_lm_mask_generator import MaskedLMMaskGenerator
from keras_nlp.models.distil_bert.distil_bert_preprocessor import (
    DistilBertPreprocessor,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras_nlp_export("keras_nlp.models.DistilBertMaskedLMPrerprocessor")
class DistilBertMaskedLMPreprocessor(DistilBertPreprocessor):
    """DistilBERT preprocessing for the masked language modeling task.

    This preprocessing layer will prepare inputs for a masked language modeling
    task. It is primarily intended for use with the
    `keras_nlp.models.DistilBertMaskedLM` task model. Preprocessing will occur in
    multiple steps.

    - Tokenize any number of input segments using the `tokenizer`.
    - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
    - Randomly select non-special tokens to mask, controlled by
      `mask_selection_rate`.
    - Construct a `(x, y, sample_weight)` tuple suitable for training with a
      `keras_nlp.models.DistilBertMaskedLM` task model.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.DistilBertMaskedLMPreprocessor.from_preset(
        "distil_bert_base_en"
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
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += ["The", "qu", "##ick", "br", "##own", "fox", "tripped"]
    vocab += ["Call", "me", "Ish", "##mael", "."]
    vocab += ["Oh", "look", "a", "whale"]
    vocab += ["I", "forgot", "my", "home", "##work"]
    tokenizer = keras_nlp.models.DistilBertTokenizer(vocabulary=vocab)
    preprocessor = keras_nlp.models.DistilBertMaskedLMPreprocessor(tokenizer)
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
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
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
