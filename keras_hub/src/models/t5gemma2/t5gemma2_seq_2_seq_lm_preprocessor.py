import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_tokenizer import T5Gemma2Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.T5Gemma2Seq2SeqLMPreprocessor")
class T5Gemma2Seq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """T5Gemma2 Seq2Seq LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.T5Gemma2Seq2SeqLM`. By default, it will take in
    batches of strings, and return outputs in a
    `(x, y, sample_weight)` format, where the `y` label is the next
    token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this
    preprocessor is attached to a `keras_hub.models.T5Gemma2Seq2SeqLM`
    instance, these methods will be called implicitly in `generate()`.

    When an `image_converter` is provided, the preprocessor also
    supports multimodal inputs with images. Images are inserted into
    the encoder sequence as placeholder tokens that the backbone's
    vision encoder will replace with image embeddings.

    Args:
        tokenizer: A `keras_hub.models.T5Gemma2Tokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
        image_converter: A `keras_hub.layers.ImageConverter` instance,
            or `None` for text-only. Defaults to `None`.
        add_start_token: If `True`, prepend the start token. Defaults
            to `False`.
        add_end_token: If `True`, append the end token. Defaults to
            `True`.
    """

    backbone_cls = T5Gemma2Backbone
    tokenizer_cls = T5Gemma2Tokenizer

    def __init__(
        self,
        tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
        image_converter=None,
        add_start_token=False,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            encoder_sequence_length=encoder_sequence_length,
            decoder_sequence_length=decoder_sequence_length,
            **kwargs,
        )
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.image_converter = image_converter

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
