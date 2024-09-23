# Copyright 2024 The KerasHub Authors
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


from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.models.bart.bart_tokenizer import BartTokenizer
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor


@keras_hub_export("keras_hub.models.BartSeq2SeqLMPreprocessor")
class BartSeq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """BART Seq2Seq LM preprocessor.

    This layer is used as preprocessor for seq2seq tasks using the BART model.
    This class subclasses `keras_hub.models.BartPreprocessor` and keeps most of
    its functionality. It has two changes from the superclass:

     1. Sets the `y` (label) and `sample_weights` fields by shifting the
        decoder input sequence one step towards the left. Both these fields are
        inferred internally, and any passed values will be ignored.
     2. Drops the last token from the decoder input sequence as it does not have
        a successor.

    Args:
        tokenizer: A `keras_hub.models.BartTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.

    Call arguments:
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Label data. Should always be `None` as the layer generates labels by
            shifting the decoder input sequence one step to the left.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights by shifting the padding mask one step to the
            left.

    Examples:

    Directly calling the layer on data
    ```python
    preprocessor = keras_hub.models.BartPreprocessor.from_preset("bart_base_en")

    # Preprocess unbatched inputs.
    inputs = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)

    # Preprocess batched inputs.
    inputs = {
        "encoder_text": ["The fox was sleeping.", "The lion was quiet."],
        "decoder_text": ["The fox was awake.", "The lion was roaring."]
    }
    preprocessor(inputs)

    # Custom vocabulary.
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

    tokenizer = keras_hub.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_hub.models.BartPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=20,
        decoder_sequence_length=10,
    )
    inputs = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.BartPreprocessor.from_preset("bart_base_en")

    # Map single sentences.
    features = {
        "encoder_text": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_text": tf.constant(
            ["The fox was awake.", "The lion was roaring."]
        )
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = BartBackbone
    tokenizer_cls = BartTokenizer

    def build(self, input_shape):
        super().build(input_shape)
        # The decoder is packed a bit differently; the format is as follows:
        # `[end_token_id, start_token_id, tokens..., end_token_id, padding...]`.
        self.decoder_packer = StartEndPacker(
            start_value=[
                self.tokenizer.end_token_id,
                self.tokenizer.start_token_id,
            ],
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
