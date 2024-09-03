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
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.preprocessor import Preprocessor


@keras_nlp_export("keras_nlp.models.MaskedLMPreprocessor")
class MaskedLMPreprocessor(Preprocessor):
    """Base class for masked language modeling preprocessing layers.

    `MaskedLMPreprocessor` tasks wrap a `keras_nlp.tokenizer.Tokenizer` to
    create a preprocessing layer for masked language modeling tasks. It is
    intended to be paired with a `keras.models.MaskedLM` task.

    All `MaskedLMPreprocessor` take inputs a single input. This can be a single
    string, a batch of strings, or a tuple of batches of string segments that
    should be combined into a single sequence. See examples below. These inputs
    will be tokenized, combined, and masked randomly along the sequence.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the masked, tokenized inputs, `y` contains the tokens
    that were masked in `x`, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    All `MaskedLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_nlp.models.MaskedLMPreprocessor.from_preset(
        "bert_base_en_uncased",
        sequence_length=256, # Optional.
    )

    # Tokenize, mask and pack a single sentence.
    x = "The quick brown fox jumped."
    x, y, sample_weight = preprocessor(x)

    # Preprocess a batch of labeled sentence pairs.
    first = ["The quick brown fox jumped.", "Call me Ishmael."]
    second = ["The fox tripped.", "Oh look, a whale."]
    x, y, sample_weight = preprocessor((first, second))

    # With a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices((first, second))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    # TODO: move common code down to this base class where possible.
