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

"""GPT2 Causal LM preprocessor layer."""

from absl import logging
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLMPreprocessor(GPT2Preprocessor):
    """GPT2 Causal LM preprocessor.

    This preprocessor is majorly used as the preprocesor for `GPT2CausalLM`.
    This class subclasses `keras_nlp.models.GPT2Preprocessor` and keeps most of
    its functionality. The only change is `GPT2CausalLMPreprocessor` sets
    `y` (label) and `sample_weights` field by shifting the input sequence one
    step towards left, and drop the last token as it does not have a successor,
    e.g., if the tokenized input is `[1, 2, 3, 0, 0]` with
    `padding_mask = [1, 1, 1, 0, 0]`, then after preprocessing, we
    will have `x = [1, 2, 3, 0]` and `y = [2, 3, 0, 0]`, with
    `padding_mask = [1, 1, 1, 0]` and `sample_weights = [1, 1, 0, 0]`.

    Args:
        tokenizer: A `keras_nlp.models.GPT2Tokenizer` instance.
        sequence_length: The length of the packed inputs.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en"
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
    """

    def call(self, x, y=None, sample_weight=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`GPT2CausalLMPreprocessor` generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored."
            )

        x = super().call(x)
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        # Target `y` will be the next token.
        y = token_ids[..., 1:]
        sample_weight = padding_mask[..., 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)
