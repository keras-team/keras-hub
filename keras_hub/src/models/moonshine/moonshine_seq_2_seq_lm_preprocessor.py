import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


@keras_hub_export("keras_hub.models.MoonshineSeq2SeqLMPreprocessor")
class MoonshineSeq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """Moonshine Seq2Seq LM preprocessor for audio-to-text tasks.

    This preprocessor converts raw audio and text inputs into a format suitable
    for the `MoonshineAudioToText` model. It processes audio waveforms into
    features using `MoonshineAudioConverter` for the encoder and tokenizes text
    using `MoonshineTokenizer` for the decoder. It supports training and
    generation.

    Args:
        audio_converter: A `MoonshineAudioConverter` instance to process audio.
        tokenizer: A `MoonshineTokenizer` instance to tokenize text.
        encoder_sequence_length: int, optional. Maximum length for audio
            features. If None, features are variable-length with padding masks.
            Defaults to None.
        decoder_sequence_length: int, optional. Maximum length for decoder token
            sequences. Defaults to 1024.
        **kwargs: Additional keyword arguments for the parent class.

    Examples:
    ```python
    # Create audio converter and tokenizer instances.
    audio_converter = keras_hub.models.MoonshineAudioConverter()
    tokenizer = keras_hub.models.MoonshineTokenizer.from_preset(
        "moonshine_base"
    )

    # Initialize the preprocessor.
    preprocessor = keras_hub.models.MoonshineSeq2SeqLMPreprocessor(
        audio_converter=audio_converter,
        tokenizer=tokenizer,
        decoder_sequence_length=8
    )

    # Prepare input data (audio tensor and text).
    inputs = {
        "audio": keras.random.normal((1, 16000, 1)),
        "text": ["the quick brown fox"]
    }

    # Process the inputs.
    preprocessed = preprocessor(inputs)
    ```
    """

    backbone_cls = MoonshineBackbone
    tokenizer_cls = MoonshineTokenizer

    def __init__(
        self,
        audio_converter,
        tokenizer,
        encoder_sequence_length=None,
        decoder_sequence_length=1024,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.audio_converter = audio_converter
        self.encoder_sequence_length = encoder_sequence_length
        self.decoder_sequence_length = decoder_sequence_length
        self.decoder_packer = None

    def build(self, input_shape):
        self.audio_converter.build(input_shape)
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        encoder_sequence_length = (
            encoder_sequence_length or self.encoder_sequence_length
        )
        decoder_sequence_length = (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )
        text = x["text"]
        audio_features = self.audio_converter(x["audio"])
        encoder_inputs = audio_features["input_values"]
        encoder_padding_mask = audio_features["attention_mask"]
        decoder_inputs = self.tokenizer(text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
        )
        x_out = {
            "encoder_input_values": encoder_inputs,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        y_out = decoder_token_ids[..., 1:]
        sample_weight_out = decoder_padding_mask[..., 1:]

        return keras.utils.pack_x_y_sample_weight(
            x_out, y_out, sample_weight_out
        )

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )
        audio_features = self.audio_converter(x["audio"])
        encoder_token_ids = audio_features["input_values"]
        encoder_padding_mask = audio_features["attention_mask"]
        decoder_text = x.get("text", [""] * keras.ops.shape(x["audio"])[0])
        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )

        return {
            "encoder_input_values": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(self, x):
        if not self.built:
            self.build(None)
        token_ids, padding_mask = (
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
        )
        ids_to_strip = self.tokenizer.special_token_ids
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)
