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


@keras_nlp_export("keras_nlp.models.TextClassifierPreprocessor")
class TextClassifierPreprocessor(Preprocessor):
    """Base class for text classification preprocessing layers.

    `TextClassifierPreprocessor` tasks wrap a `keras_nlp.tokenizer.Tokenizer` to
    create a preprocessing layer for text classification tasks. It is intended
    to be paired with a `keras_nlp.models.TextClassifier` task.

    All `TextClassifierPreprocessor` take inputs three ordered inputs, `x`, `y`,
    and `sample_weight`. `x`, the first input, should always be included. It can
    be a single string, a batch of strings, or a tuple of batches of string
    segments that should be combined into a single sequence. See examples below.
    `y` and `sample_weight` are optional inputs that will be passed through
    unaltered. Usually, `y` will be the classification label, and
    `sample_weight` will not be provided.

    The layer will output either `x`, an `(x, y)` tuple if labels were provided,
    or an `(x, y, sample_weight)` tuple if labels and sample weight were
    provided. `x` will be a dictionary with tokenized input, the exact contents
    of the dictionary will depend on the model being used.

    All `TextClassifierPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_nlp.models.TextClassifierPreprocessor.from_preset(
        "bert_base_en_uncased",
        sequence_length=256, # Optional.
    )

    # Tokenize and pad/truncate a single sentence.
    x = "The quick brown fox jumped."
    x = preprocessor(x)

    # Tokenize and pad/truncate a labeled sentence.
    x, y = "The quick brown fox jumped.", 1
    x, y = preprocessor(x, y)

    # Tokenize and pad/truncate a batch of labeled sentences.
    x, y = ["The quick brown fox jumped.", "Call me Ishmael."], [1, 0]
    x, y = preprocessor(x, y)

    # Tokenize and combine a batch of labeled sentence pairs.
    first = ["The quick brown fox jumped.", "Call me Ishmael."]
    second = ["The fox tripped.", "Oh look, a whale."]
    labels = [1, 0]
    x, y = (first, second), labels
    x, y = preprocessor(x, y)

    # Use a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices(((first, second), labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    # TODO: move common code down to this base class where possible.
