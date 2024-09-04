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


@keras_nlp_export("keras_nlp.models.CausalLMPreprocessor")
class CausalLMPreprocessor(Preprocessor):
    """Base class for causal language modeling preprocessing layers.

    `CausalLMPreprocessor` tasks wrap a `keras_nlp.tokenizer.Tokenizer` to
    create a preprocessing layer for causal language modeling tasks. It is
    intended to be paired with a `keras.models.CausalLM` task.

    All `CausalLMPreprocessor` take inputs a single input. This can be a single
    string or a batch of strings. See examples below. These inputs
    will be tokenized and padded/truncated to a fixed sequence length.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
    offset by 1, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    a `CausalLMPreprocessor` contains two extra methods, `generate_preprocess`
    and `generate_postprocess` for use with generation. See examples below.

    All `CausalLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_nlp.models.CausalLMPreprocessor.from_preset(
        "bert_base_en_uncased",
        sequence_length=256, # Optional.
    )

    # Tokenize, mask and pack a single sentence.
    x = "The quick brown fox jumped."
    x, y, sample_weight = preprocessor(x)

    # Tokenize and pad/truncate a batch of labeled sentences.
    x = ["The quick brown fox jumped.", "Call me Ishmael."]
    x, y, sample_weight = preprocessor(x)

    # With a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Generate preprocess and postprocess.
    x = preprocessor.generate_preprocess(x)  # Tokenized numeric inputs.
    x = preprocessor.generate_postprocess(x)  # Detokenized string outputs.
    ```
    """

    # TODO: move common code down to this base class where possible.
