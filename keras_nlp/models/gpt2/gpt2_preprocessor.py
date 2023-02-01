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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2Preprocessor(Preprocessor):
    """GPT2 preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do 2 things:

    - Tokenize the input using the `tokenizer`.
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
        add_start_token: If true, the preprocessor will append the tokenizer
            start token to each input sequence.
        add_end_token: If true, the preprocessor will append the tokenizer
            end token to each input sequence.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.GPT2Preprocessor.from_preset("gpt2_base_en")

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco!"])

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

    # Alternatively, you can create a preprocessor from your own vocabulary.
    # The usage is exactly the same as above.
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_nlp.models.GPT2Tokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.GPT2Preprocessor(
        tokenizer=tokenizer,
        sequence_length=20,
    )
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length,
        add_start_token=False,
        add_end_token=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        x = convert_inputs_to_list_of_tensor_segments(x)
        if len(x) > 1:
            raise ValueError(
                "GPT2 requires each input feature to contain only "
                f"one segment, but received {len(x)}. If you are using GPT2 "
                "for a multi-segment classification task, please refer to "
                "classification models like BERT or RoBERTa."
            )
        token_ids = self.tokenizer(x[0])
        input_is_1d = len(token_ids.shape) == 1
        if input_is_1d:
            token_ids = tf.RaggedTensor.from_tensor([token_ids])
        if self.add_start_token:
            start_tokens = tf.fill(
                [tf.shape(token_ids)[0], 1],
                self.tokenizer.start_token_id,
            )
            token_ids = tf.concat([start_tokens, token_ids], axis=1)
        if self.add_end_token:
            end_tokens = tf.fill(
                [tf.shape(token_ids)[0], 1],
                self.tokenizer.end_token_id,
            )
            token_ids = tf.concat([token_ids, end_tokens], axis=1)
        mask = tf.ones_like(token_ids, dtype=tf.bool)
        shape_after_padding = tf.stack(
            [tf.constant(-1), self.sequence_length],
            axis=0,
        )
        mask = mask.to_tensor(shape=shape_after_padding)
        token_ids = token_ids.to_tensor(
            shape=shape_after_padding,
            default_value=self.tokenizer.pad_token_id,
        )
        if input_is_1d:
            # If the input is a single string, we let the output be a 1D tensor.
            token_ids = tf.squeeze(token_ids, axis=0)
            mask = tf.squeeze(mask, axis=0)
        x = {
            "token_ids": token_ids,
            "padding_mask": mask,
        }

        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def tokenizer_cls(cls):
        return GPT2Tokenizer
