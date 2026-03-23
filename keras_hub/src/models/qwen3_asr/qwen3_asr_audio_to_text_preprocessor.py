import keras
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.audio_to_text_preprocessor import (
    AudioToTextPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import (
    Qwen3ASRBackbone,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import (
    Qwen3ASRTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.Qwen3ASRAudioToTextPreprocessor")
class Qwen3ASRAudioToTextPreprocessor(AudioToTextPreprocessor):
    """Qwen3-ASR preprocessor for audio-to-text tasks.

    This preprocessor converts raw audio and text inputs into a format
    suitable for the ``Qwen3ASRAudioToText`` model. Audio waveforms are
    converted to mel spectrograms via ``Qwen3ASRAudioConverter``, and text
    is tokenized via ``Qwen3ASRTokenizer``.

    The Qwen3-ASR architecture uses **replacement embedding**: the
    preprocessor constructs ``token_ids`` that contain ``<|AUDIO|>``
    placeholder tokens at the positions where the audio encoder output
    will be scattered in. The sequence looks like::

        [<|AUDIO|>] * num_audio_tokens + [text_tokens] + [padding]

    Args:
        audio_converter: A ``Qwen3ASRAudioConverter`` instance.
        tokenizer: A ``Qwen3ASRTokenizer`` instance.
        audio_token_id: int. Token ID for the audio placeholder.
            Defaults to ``151676``.
        decoder_sequence_length: int. Maximum length for the text portion
            of the token sequence. Defaults to ``448``.
        **kwargs: Additional keyword arguments for the parent class.

    Examples:
    ```python
    converter = keras_hub.layers.Qwen3ASRAudioConverter()
    tokenizer = keras_hub.models.Qwen3ASRTokenizer.from_preset(
        "qwen3_asr_1.7b"
    )
    preprocessor = keras_hub.models.Qwen3ASRAudioToTextPreprocessor(
        audio_converter=converter,
        tokenizer=tokenizer,
        decoder_sequence_length=8,
    )
    inputs = {
        "audio": np.random.normal(size=(1, 16000)),
        "text": ["hello world"],
    }
    x, y, sample_weight = preprocessor(inputs)
    ```
    """

    backbone_cls = Qwen3ASRBackbone
    tokenizer_cls = Qwen3ASRTokenizer

    def __init__(
        self,
        audio_converter,
        tokenizer,
        audio_token_id=151676,
        decoder_sequence_length=448,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.audio_converter = audio_converter
        self.audio_token_id = audio_token_id
        self.decoder_sequence_length = decoder_sequence_length
        self.decoder_packer = None
        self._special_token_ids_set = None

    def build(self, input_shape):
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self._special_token_ids_set = set(self.tokenizer.special_token_ids)
        if self.tokenizer.pad_token_id is not None:
            self._special_token_ids_set.add(self.tokenizer.pad_token_id)
        # Also exclude the audio placeholder from post-processing.
        self._special_token_ids_set.add(self.audio_token_id)
        self.built = True

    def _compute_num_audio_tokens(self, audio_features):
        """Compute how many audio tokens the encoder will produce."""
        audio_time = keras.ops.shape(audio_features)[1]
        return (audio_time + 7) // 8

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

        # Convert audio to mel spectrogram features.
        audio_features = self.audio_converter(x["audio"])
        num_audio_tokens = self._compute_num_audio_tokens(audio_features)
        batch_size = keras.ops.shape(audio_features)[0]

        # Tokenize text and pack with end token.
        text = x["text"]
        decoder_inputs = self.tokenizer(text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
            add_end_value=True,
        )

        # Teacher forcing: input is [:-1], target is [1:].
        text_token_ids = decoder_token_ids[..., :-1]
        text_padding_mask = decoder_padding_mask[..., :-1]

        # Build full token_ids: [audio_placeholders | text_tokens].
        audio_placeholders = keras.ops.full(
            (batch_size, num_audio_tokens),
            self.audio_token_id,
            dtype="int32",
        )
        token_ids = keras.ops.concatenate(
            [audio_placeholders, text_token_ids], axis=-1
        )

        # Build full padding mask.
        audio_mask = keras.ops.ones(
            (batch_size, num_audio_tokens), dtype="int32"
        )
        padding_mask = keras.ops.concatenate(
            [audio_mask, text_padding_mask], axis=-1
        )

        x_out = {
            "audio_features": audio_features,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
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
        num_audio_tokens = self._compute_num_audio_tokens(audio_features)
        audio_batch_size = keras.ops.shape(x["audio"])[0]

        decoder_text = x.get("text", None)
        if decoder_text is None:
            decoder_token_ids = [[self.tokenizer.pad_token_id]] * (
                audio_batch_size
            )
        else:
            if isinstance(decoder_text, str):
                decoder_text = [decoder_text] * audio_batch_size
            elif len(decoder_text) != audio_batch_size:
                if len(decoder_text) == 1:
                    decoder_text = decoder_text * audio_batch_size
                else:
                    raise ValueError(
                        f"Batch size mismatch between audio "
                        f"({audio_batch_size}) and text prompts "
                        f"({len(decoder_text)})"
                    )
            decoder_token_ids = self.tokenizer(decoder_text)

        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )

        # Build full token_ids: [audio_placeholders | text_tokens].
        batch_size = keras.ops.shape(decoder_token_ids)[0]
        audio_placeholders = keras.ops.full(
            (batch_size, num_audio_tokens),
            self.audio_token_id,
            dtype="int32",
        )
        token_ids = keras.ops.concatenate(
            [audio_placeholders, decoder_token_ids], axis=-1
        )

        audio_mask = keras.ops.ones(
            (batch_size, num_audio_tokens), dtype="int32"
        )
        padding_mask = keras.ops.concatenate(
            [audio_mask, decoder_padding_mask], axis=-1
        )

        return {
            "audio_features": audio_features,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(self, x):
        if not self.built:
            self.build(None)
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids_np = keras.ops.convert_to_numpy(token_ids)
        padding_mask_np = keras.ops.convert_to_numpy(padding_mask)
        vocab_size = self.tokenizer.vocabulary_size()
        processed_sequences = []
        for i in range(token_ids_np.shape[0]):
            sequence = token_ids_np[i]
            mask = padding_mask_np[i].astype(bool)
            valid_tokens = sequence[mask]
            filtered_tokens = [
                int(token)
                for token in valid_tokens
                if token not in self._special_token_ids_set
                and 0 <= token < vocab_size
            ]
            processed_sequences.append(filtered_tokens)
        processed_sequences = tf.ragged.constant(
            processed_sequences, dtype=tf.int32
        )
        return self.tokenizer.detokenize(processed_sequences)
