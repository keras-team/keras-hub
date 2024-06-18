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


import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_nlp.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_nlp.src.models.preprocessor import Preprocessor
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)


@keras_nlp_export("keras_nlp.models.GemmaPreprocessor")
class GemmaPreprocessor(Preprocessor):
    """Gemma preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do 2 things:

    - Tokenize the inputs using the `tokenizer`.
    - Construct a dictionary with keys `"token_ids"`, `"padding_mask"`, that can
        be passed directly to a `keras_nlp.models.GemmaBackbone`.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    The call method of this layer accepts three arguments, `x`, `y`, and
    `sample_weight`. `x` can be a python string or tensor representing a single
    segment, a list of python strings representing a batch of single segments,
    or a list of tensors representing multiple segments to be packed together.
    `y` and `sample_weight` are both optional, can have any format, and will be
    passed through unaltered.

    `GemmaPreprocessor` expects the input to have only one segment, as Gemma is
    mainly used for generation tasks. For tasks having multi-segment inputs
    please combine inputs into a single string input before passing to the
    preprocessor layer.

    Args:
        tokenizer: A `keras_nlp.models.GemmaTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
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
    preprocessor = keras_nlp.models.GemmaPreprocessor.from_preset(
        "gemma_2b_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=8,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
    )
    tokenizer = keras_nlp.models.GemmaTokenizer(
        proto=bytes_io.getvalue(),
    )
    preprocessor = keras_nlp.models.GemmaPreprocessor(tokenizer=tokenizer)
    preprocessor("The quick brown fox jumped.")
    ```

    Apply preprocessing to a `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.GemmaPreprocessor.from_preset(
        "gemma_2b_en"
    )

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

    tokenizer_cls = GemmaTokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=8192,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

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
                "GemmaPreprocessor requires each input to contain only "
                f"one segment, but received {len(x)}. If you are using Gemma "
                "for a multi-segment classification task, please combine your "
                "input into a single string."
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
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

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
