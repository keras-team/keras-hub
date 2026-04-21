import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.audio_to_text_preprocessor import (
    AudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


@keras_hub_export("keras_hub.models.WhisperAudioToTextPreprocessor")
class WhisperAudioToTextPreprocessor(AudioToTextPreprocessor):
    """Whisper audio-to-text preprocessor.

    This preprocessor converts raw audio and text inputs into a format suitable
    for the `keras_hub.models.WhisperAudioToText` model. Audio waveforms are
    converted to log-mel spectrograms using `WhisperAudioConverter`, and text
    is tokenized with `WhisperTokenizer` for the decoder. It supports both
    training (producing `(x, y, sample_weight)` tuples) and generation
    (producing a dictionary of model inputs).

    Args:
        audio_converter: A `keras_hub.layers.WhisperAudioConverter` instance.
        tokenizer: A `keras_hub.models.WhisperTokenizer` instance.
        decoder_sequence_length: int. The maximum length of the packed decoder
            token sequence. Defaults to `448`, matching Whisper's
            `max_decoder_sequence_length`.

    Examples:
    ```python
    preprocessor = keras_hub.models.WhisperAudioToTextPreprocessor.from_preset(
        "whisper_tiny_en"
    )

    # Process a single audio-text pair for training.
    inputs = {
        "audio": keras.random.normal((1, 16000)),
        "text": ["the quick brown fox"],
    }
    x, y, sample_weight = preprocessor(inputs)

    # Prepare inputs for generation with an optional decoder prompt.
    gen_inputs = preprocessor.generate_preprocess({
        "audio": keras.random.normal((1, 16000)),
        "text": ["the quick"],
    })
    ```
    """

    backbone_cls = WhisperBackbone
    tokenizer_cls = WhisperTokenizer
    audio_converter_cls = WhisperAudioConverter

    def __init__(
        self,
        audio_converter,
        tokenizer,
        decoder_sequence_length=448,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.audio_converter = audio_converter
        self.decoder_sequence_length = decoder_sequence_length
        self.decoder_packer = None

    def build(self, input_shape):
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.bos_token_id,
            end_value=self.tokenizer.eos_token_id,
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
        encoder_features = self.audio_converter(x["audio"])
        decoder_inputs = self.tokenizer(x["text"])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
            add_end_value=True,
        )
        x_out = {
            "encoder_features": encoder_features,
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
        """Convert raw audio (and an optional text prompt) to model inputs.

        This mirrors `call()`, but does not append the end token to the decoder
        prompt and does not produce labels. If `x` contains no `"text"` key, the
        decoder prompt is initialized with just the `bos` token so generation
        starts from scratch.
        """
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )
        encoder_features = self.audio_converter(x["audio"])
        audio_batch_size = keras.ops.shape(encoder_features)[0]
        decoder_text = x.get("text", None) if isinstance(x, dict) else None
        if decoder_text is None:
            decoder_token_ids = [[self.tokenizer.bos_token_id]] * int(
                audio_batch_size
            )
        else:
            if isinstance(decoder_text, str):
                decoder_text = [decoder_text] * int(audio_batch_size)
            decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )
        return {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(self, x):
        """Convert generated token ids back to strings."""
        if not self.built:
            self.build(None)
        token_ids, padding_mask = (
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
        )
        ids_to_strip = self.tokenizer.special_token_ids
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    @property
    def decoder_sequence_length(self):
        """The padded length of decoder input sequences."""
        return self._decoder_sequence_length

    @decoder_sequence_length.setter
    def decoder_sequence_length(self, value):
        self._decoder_sequence_length = value
        if self.decoder_packer is not None:
            self.decoder_packer.sequence_length = value

    @property
    def sequence_length(self):
        """Alias for `decoder_sequence_length`."""
        return self.decoder_sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self.decoder_sequence_length = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_sequence_length": self.decoder_sequence_length,
            }
        )
        return config
