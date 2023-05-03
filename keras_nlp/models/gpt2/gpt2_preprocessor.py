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

"""GPT2 preprocessor layer."""

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.start_end_packer import StartEndPacker
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.GPT2Preprocessor")
class GPT2Preprocessor(Preprocessor):
    """GPT2 preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do 2 things:

    - Tokenize the inputs using the `tokenizer`.
    - Construct a dictionary with keys `"token_ids"`, `"padding_mask"`, that can
        be passed directly to a `keras_nlp.models.GPT2Backbone`.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    The call method of this layer accepts three arguments, `x`, `y`, and
    `sample_weight`. `x` can be a python string or tensor representing a single
    segment, a list of python strings representing a batch of single segments,
    or a list of tensors representing multiple segments to be packed together.
    `y` and `sample_weight` are both optional, can have any format, and will be
    passed through unaltered.

    `GPT2Preprocessor` forces the input to have only one segment, as GPT2 is
    mainly used for generation tasks. For tasks having multi-segment inputs
    like "glue/mnli", please use a model designed for classification purposes
    such as BERT or RoBERTa.

    Args:
        tokenizer: A `keras_nlp.models.GPT2Tokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If true, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If true, the preprocessor will append the tokenizer
            end token to each input sequence.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_nlp.models.GPT2Preprocessor.from_preset("gpt2_base_en")

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of single sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Custom vocabulary.
    features = ["a quick fox.", "a fox quick."]
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_nlp.models.GPT2Tokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.GPT2Preprocessor(tokenizer=tokenizer)
    preprocessor("The quick brown fox jumped.")
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.GPT2Preprocessor.from_preset("gpt2_base_en")

    text = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
    label = tf.constant([1, 1])

    # Map labeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices((text, label))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(text)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.packer = StartEndPacker(
            start_value=tokenizer.start_token_id,
            end_value=tokenizer.end_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
            return_padding_mask=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        x = convert_inputs_to_list_of_tensor_segments(x)
        if len(x) != 1:
            raise ValueError(
                "GPT2 requires each input feature to contain only "
                f"one segment, but received {len(x)}. If you are using GPT2 "
                "for a multi-segment classification task, please refer to "
                "classification models like BERT or RoBERTa."
            )
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask = self.packer(
            self.tokenizer(x[0]),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def tokenizer_cls(cls):
        return GPT2Tokenizer
