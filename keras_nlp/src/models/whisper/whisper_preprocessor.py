# Copyright 2023 The KerasNLP Authors
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
from absl import logging

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_nlp.src.models.preprocessor import Preprocessor
from keras_nlp.src.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)


@keras_nlp_export("keras_nlp.models.WhisperPreprocessor")
class WhisperPreprocessor(Preprocessor):
    """A Whisper preprocessing layer which handles audio and text input.

    This preprocessing layer will do three things:

     1. Compute the log-mel spectrogram of the audio tensor inputs using
        `audio_feature_extractor`.
     2. Tokenize decoder inputs using the `tokenizer`.
     2. Add the appropriate special tokens - `"<|startoftranscript|>", task
        token, language token, `"<|endoftext|>"`, etc.
     3. Construct a dictionary with keys `"encoder_features"`,
        `"decoder_token_ids"`, `"decoder_padding_mask"` that can be passed
        directly to a Whisper model.

    Args:
        tokenizer: A `keras_nlp.models.WhisperTokenizer` instance.
        audio_feature_extractor: A
            `keras_nlp.models.WhisperAudioFeatureExtractor` instance or `None`.
            If `None` a feature extractor with default parameters will be
            created.
        decoder_sequence_length: The length of the packed decoder inputs.
        language: string, language token. Should only be passed if your
            tokenizer is multilingual.
        task: string, task name. One of `"transcribe"`, `"translate"`. Should
            only be passed if your tokenizer is multilingual.
        no_timestamps: bool. If True, `"<|no_timestamps|>"` will be added as a
            special token to your input.

    Call arguments:
        x: A dictionary with `"encoder_audio"` and `"decoder_text"` as its keys.
            `"encoder_audio"` should correspond to the input audio tensor.
            `"decoder_text"` should be a tensor of single string sequences.
            Inputs may be batched or unbatched. Raw python inputs will be
            converted to tensors.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_nlp.models.WhisperPreprocessor.from_preset(
        "whisper_tiny_en",
    )

    # Preprocess unbatched inputs.
    input_data = {
        "encoder_audio": tf.ones((200,)),
        "decoder_text": "The quick brown fox jumped.",
    }
    preprocessor(input_data)

    # Preprocess batched inputs.
    input_data = {
        "encoder_audio": tf.ones((2, 200)),
        "decoder_text": ["The quick brown fox jumped.", "Call me Ishmael."],
    }
    preprocessor(input_data)

    # Custom audio feature extractor and vocabulary.
    audio_feature_extractor = keras_nlp.models.WhisperAudioFeatureExtractor(
        num_mels=80,
        num_fft_bins=400,
        stride=100,
        sampling_rate=100,
        max_audio_length=5,
    )

    features = ["a quick fox.", "a fox quick."]
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    special_tokens = {
        "<|startoftranscript|>": 9,
        "<|endoftext|>": 10,
        "<|notimestamps|>": 11,
        "<|transcribe|>": 12,
        "<|translate|>": 13,
    }

    tokenizer = keras_nlp.models.WhisperTokenizer(
        vocabulary=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )
    preprocessor = keras_nlp.models.WhisperPreprocessor(
        audio_feature_extractor=audio_feature_extractor,
        tokenizer=tokenizer,
    )

    input_data = {
        "encoder_audio": tf.ones((200,)),
        "decoder_text": "The quick brown fox jumped.",
    }
    preprocessor(input_data)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.WhisperPreprocessor.from_preset(
        "whisper_tiny_en")

    # Map labeled single sentences.
    features = {
        "encoder_audio": tf.ones((2, 200)),
        "decoder_text": ["The quick brown fox jumped.", "Call me Ishmael."],
    }
    labels = tf.constant(["True", "False"])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
    features = {
        "encoder_audio": tf.ones((2, 200)),
        "decoder_text": ["The quick brown fox jumped.", "Call me Ishmael."],
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    tokenizer_cls = WhisperTokenizer

    def __init__(
        self,
        tokenizer,
        audio_feature_extractor=None,
        decoder_sequence_length=448,
        language=None,
        task=None,
        no_timestamps=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if audio_feature_extractor is None:
            audio_feature_extractor = WhisperAudioFeatureExtractor()
        self.audio_feature_extractor = audio_feature_extractor
        self.tokenizer = tokenizer
        self.decoder_packer = None
        self.decoder_sequence_length = decoder_sequence_length
        self.language = language
        self.task = task
        self.no_timestamps = no_timestamps

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.

        # Create list of tokens to be prepended to decoder inputs.
        bos_tokens = [self.tokenizer.bos_token_id]
        if self.tokenizer.language_tokens is not None:
            if (
                self.language is None
                or self.language not in self.tokenizer.language_tokens
            ):
                raise ValueError(
                    "You must pass a non-None value for `language` when using "
                    "a multilingual tokenizer. The value must be one of "
                    f'{",".join(self.tokenizer.language_tokens.keys())}. '
                    f"Received: language={self.language}."
                )
            if self.task is None or self.task not in [
                "transcribe",
                "translate",
            ]:
                raise ValueError(
                    "You must pass a non-None value for `task` when using "
                    "a multilingual tokenizer. The value must be one of "
                    '`"transcribe"`, `"translate"`. '
                    f"Received: task={self.task}."
                )

            bos_tokens += [self.tokenizer.language_tokens[self.language]]

            if self.task == "transcribe":
                bos_tokens += [self.tokenizer.special_tokens["<|transcribe|>"]]
            elif self.task == "translate":
                bos_tokens += [self.tokenizer.special_tokens["<|translate|>"]]
        else:
            if self.language is not None:
                logging.info(
                    "`tokenizer` is monolingual, and `language` has a "
                    "non-`None` value. Setting `language` to `None`."
                )
                self.language = None
            if self.task is not None:
                logging.info(
                    "`tokenizer` is monolingual, and `task` has a "
                    "non-`None` value. Setting `task` to `None`."
                )
                self.task = None

        if self.no_timestamps:
            bos_tokens += [self.tokenizer.no_timestamps_token_id]

        # TODO: Use `MultiSegmentPacker` instead of `StartEndPacker` once we
        # want to move to multi-segment packing and have improved
        # `MultiSegmentPacker`'s performance.
        self.decoder_packer = StartEndPacker(
            start_value=bos_tokens,
            end_value=self.tokenizer.eos_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )

    def call(self, x, y=None, sample_weight=None, decoder_sequence_length=None):
        if not (
            isinstance(x, dict)
            and ["encoder_audio", "decoder_text"] == list(x.keys())
        ):
            raise ValueError(
                '`x` must be a dictionary, containing the keys `"encoder_audio"`'
                f' and `"decoder_text"`. Received x={x}.'
            )

        encoder_audio = x["encoder_audio"]
        decoder_text = x["decoder_text"]

        encoder_audio = convert_inputs_to_list_of_tensor_segments(encoder_audio)
        decoder_text = convert_inputs_to_list_of_tensor_segments(decoder_text)

        if len(encoder_audio) > 1 or len(decoder_text) > 1:
            raise ValueError(
                '`WhisperPreprocessor` requires both `"encoder_audio"` and '
                f'`"decoder_text"` to contain only one segment, but received '
                f"{len(encoder_audio)} and {len(decoder_text)}, respectively."
            )

        encoder_features = self.audio_feature_extractor(encoder_audio[0])
        decoder_sequence_length = (
            decoder_sequence_length or self.decoder_sequence_length
        )
        decoder_inputs = self.tokenizer(decoder_text[0])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length,
        )

        x = {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "audio_feature_extractor": keras.layers.serialize(
                    self.audio_feature_extractor
                ),
                "decoder_sequence_length": self.decoder_sequence_length,
                "language": self.language,
                "task": self.task,
                "no_timestamps": self.no_timestamps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])

        if "audio_feature_extractor" in config and isinstance(
            config["audio_feature_extractor"], dict
        ):
            config["audio_feature_extractor"] = keras.layers.deserialize(
                config["audio_feature_extractor"]
            )

        return cls(**config)

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
