import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_tokenizer import T5GemmaTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.T5GemmaSeq2SeqLMPreprocessor")
class T5GemmaSeq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """T5Gemma Seq2Seq LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.T5GemmaSeq2SeqLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.T5GemmaSeq2SeqLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.T5GemmaTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
        add_start_token: If `True`, the preprocessor will prepend the
            tokenizer start token to each input sequence. For T5Gemma models,
            this should be `False`. Defaults to `False`.
        add_end_token: If `True`, the preprocessor will append the tokenizer end
            token to each input sequence. For T5Gemma models, this should be
            `True`. Defaults to `True`.

    Call arguments:
        x: A dictionary with two keys, `"encoder_text"` and `"decoder_text"`.
           The values can be a string, a `tf.Tensor` or a list of python
           strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        encoder_sequence_length: Pass to override the configured
            `encoder_sequence_length` of the layer.
        decoder_sequence_length: Pass to override the configured
            `decoder_sequence_length` of the layer.

    Examples:
    ```python
    import tensorflow as tf
    import numpy as np

    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )

    # For example usage, see the dictionary example below which provides
    # both encoder and decoder text.
    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
    # Tokenize a dictionary with separate encoder and decoder inputs.
    preprocessor({
        "encoder_text": "The quick brown fox jumped.",
        "decoder_text": "The fast fox."
    })

    # Apply tokenization to a `tf.data.Dataset`.
    encoder_features = tf.constant(["The quick brown fox.", "Call me Ishmael."])
    decoder_features = tf.constant(["The fast fox.", "I am Ishmael."])
    ds = tf.data.Dataset.from_tensor_slices(
        {"encoder_text": encoder_features, "decoder_text": decoder_features}
    )
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare tokens for generation.
    preprocessor.generate_preprocess({
        "encoder_text": "The quick brown fox jumped.",
        "decoder_text": "The fast fox."
    })

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        'decoder_token_ids': np.array([[2, 714, 4320, 8426, 25341, 1, 0, 0]]),
        'decoder_padding_mask': np.array([[1, 1, 1, 1, 1, 1, 0, 0]]),
    })
    ```
    """

    backbone_cls = T5GemmaBackbone
    tokenizer_cls = T5GemmaTokenizer

    def __init__(
        self,
        tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
        add_start_token=False,
        add_end_token=True,
        **kwargs,
    ):
        # Do not pass `add_start_token` and `add_end_token` to the base class.
        super().__init__(
            tokenizer=tokenizer,
            encoder_sequence_length=encoder_sequence_length,
            decoder_sequence_length=decoder_sequence_length,
            **kwargs,
        )
        # Store them directly on the subclass instance.
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        encoder_inputs = self.tokenizer(x["encoder_text"])
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_inputs,
            sequence_length=encoder_sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        decoder_inputs = self.tokenizer(x["decoder_text"])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
            add_start_value=True,
            add_end_value=self.add_end_token,
        )
        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)

        if isinstance(x, dict):
            encoder_text = x["encoder_text"]
            decoder_text = x["decoder_text"]
        else:
            encoder_text = x
            decoder_text = tf.fill((tf.shape(encoder_text)[0],), "")

        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        encoder_token_ids = self.tokenizer(encoder_text)
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_token_ids,
            sequence_length=None,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_start_value=True,
            add_end_value=False,
        )

        return {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config
