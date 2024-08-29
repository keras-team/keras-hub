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
from keras_nlp.src.models.causal_lm_preprocessor import CausalLMPreprocessor


@keras_nlp_export("keras_nlp.models.Seq2SeqLMPreprocessor")
class Seq2SeqLMPreprocessor(CausalLMPreprocessor):
    """Base class for seq2seq language modeling preprocessing layers.

    `Seq2SeqLMPreprocessor` tasks wrap a `keras_nlp.tokenizer.Tokenizer` to
    create a preprocessing layer for seq2seq language modeling tasks. It is
    intended to be paired with a `keras.models.Seq2SeqLM` task.

    All `Seq2SeqLMPreprocessor` layers take inputs a dictionary input with keys
    `"encoder_text"` and `"decoder_text"`.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
    offset by 1, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    a `Seq2SeqLMPreprocessor` contains two extra methods, `generate_preprocess`
    and `generate_postprocess` for use with generation. See examples below.

    All `Seq2SeqLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_nlp.models.Seq2SeqLMPreprocessor.from_preset(
        "bart_base_en",
        encoder_sequence_length=256,
        decoder_sequence_length=256,
    )

    # Tokenize, mask and pack a single sentence.
    x = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake.",
    }
    x, y, sample_weight = preprocessor(x)

    # Tokenize and pad/truncate a batch of labeled sentences.
    x = {
        "encoder_text": ["The fox was sleeping."],
        "decoder_text": ["The fox was awake."],
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
