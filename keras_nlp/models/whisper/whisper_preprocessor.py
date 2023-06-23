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
"""Whisper preprocessor layer."""

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.start_end_packer import StartEndPacker
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.models.whisper.whisper_presets import backbone_presets
from keras_nlp.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.WhisperPreprocessor")
class WhisperPreprocessor(Preprocessor):
    """A Whisper preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     1. Tokenize both encoder inputs and decoder inputs using the `tokenizer`.
        Both inputs can contain only one segment.
     2. Add the appropriate special tokens - `"<s>"`, `"</s>"` and `"<pad>"`.
     3. Construct a dictionary with keys `"encoder_token_ids"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"`, `"decoder_padding_mask"`
        that can be passed directly to a Whisper model.

    Args:
        tokenizer: A `keras_nlp.models.WhisperTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.

    Call arguments:
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_nlp.models.WhisperPreprocessor.from_preset("whisper_base_en")

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

    tokenizer = keras_nlp.models.WhisperTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.WhisperPreprocessor(
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
    preprocessor = keras_nlp.models.WhisperPreprocessor.from_preset("whisper_base_en")

    # Map labeled single sentences.
    features = {
        "encoder_text": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_text": tf.constant(
            ["The fox was awake.", "The lion was silent."]
        )
    }
    labels = tf.constant(["True", "False"])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
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

    def __init__(
        self,
        audio_feature_extractor,
        tokenizer,
        decoder_sequence_length=448,
        language=None,
        task=None,
        no_timestamps=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_feature_extractor = audio_feature_extractor
        self.tokenizer = tokenizer
        self.language = language
        self.task = task
        self.no_timestamps = no_timestamps

        # The decoder is packed a bit differently; the format is as follows:
        # `[end_token_id, start_token_id, tokens..., end_token_id, padding...]`.
        bos_tokens = [self.tokenizer.bos_token_id]
        if self.tokenizer.language_tokens is not None:
            if language is not None:
                bos_tokens += [self.tokenizer.language_tokens[language]]
            if task == "transcribe":
                bos_tokens += [self.tokenizer.task_tokens["<|transcribe|>"]]
            elif task == "translate":
                bos_tokens += [self.tokenizer.task_tokens["<|translate|>"]]

        if no_timestamps:
            bos_tokens += [self.tokenizer.no_timestamps_token_id]

        # TODO: Use `MultiSegmentPacker` instead of `StartEndPacker` once we
        # want to move to multi-segment packing and have improved
        # `MultiSegmentPacker`'s performance.
        self.decoder_packer = StartEndPacker(
            start_value=bos_tokens,
            end_value=self.tokenizer.eos_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=decoder_sequence_length,
            return_padding_mask=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_sequence_length": self.decoder_packer.sequence_length,
                "language": self.language,
                "task": self.task,
                "no_timestamps": self.no_timestamps,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
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

        decoder_inputs = self.tokenizer(decoder_text[0])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs
        )

        x = {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def audio_feature_extractor_cls(cls):
        return WhisperAudioFeatureExtractor

    @classproperty
    def tokenizer_cls(cls):
        return WhisperTokenizer

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        # Override base class's `from_preset` to handle audio feature extractor
        # and `decoder_sequence_length`.
        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        audio_feature_extractor = cls.audio_feature_extractor_cls.from_preset(
            preset
        )
        tokenizer = cls.tokenizer_cls.from_preset(preset)

        metadata = cls.presets[preset]
        # For task model presets, the backbone config is nested.
        if "backbone" in metadata["config"]:
            backbone_config = metadata["config"]["backbone"]["config"]
        else:
            backbone_config = metadata["config"]

        # Use model's `max_decoder_sequence_length` if `decoder_sequence_length`
        # is unspecified; otherwise check that `decoder_sequence_length` is not
        # too long.
        decoder_sequence_length = kwargs.pop("decoder_sequence_length", None)
        max_decoder_sequence_length = backbone_config[
            "max_decoder_sequence_length"
        ]

        def check_sequence_length(sequence_length, max_sequence_length, name):
            if sequence_length is not None:
                if sequence_length > max_sequence_length:
                    raise ValueError(
                        f"`{name}` cannot be longer than `{preset}` "
                        f"preset's `max_{name}` of {max_sequence_length}. "
                        f"Received: {sequence_length}."
                    )
                return sequence_length
            else:
                return max_sequence_length

        decoder_sequence_length = check_sequence_length(
            decoder_sequence_length,
            max_decoder_sequence_length,
            "decoder_sequence_length",
        )

        return cls(
            audio_feature_extractor=audio_feature_extractor,
            tokenizer=tokenizer,
            decoder_sequence_length=decoder_sequence_length,
            **kwargs,
        )
