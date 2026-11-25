import keras
import numpy as np

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
from keras_hub.src.utils.tensor_utils import strip_to_ragged_python
from keras_hub.src.utils.tensor_utils import tf

TASKS = ("transcribe", "translate")


@keras_hub_export("keras_hub.models.WhisperAudioToTextPreprocessor")
class WhisperAudioToTextPreprocessor(AudioToTextPreprocessor):
    """Whisper audio-to-text preprocessor.

    This preprocessor converts raw audio and text inputs into a format suitable
    for the `keras_hub.models.WhisperAudioToText` model. Audio waveforms are
    converted to log-mel spectrograms using `WhisperAudioConverter`, and text
    is tokenized with `WhisperTokenizer` for the decoder. It supports both
    training (producing `(x, y, sample_weight)` tuples) and generation
    (producing a dictionary of model inputs).

    Every decoder sequence starts with Whisper's control prefix. On
    English-only checkpoints (a tokenizer without `language_tokens`) the
    prefix is `<|startoftranscript|>` followed by `<|notimestamps|>`. On
    multilingual checkpoints it is `<|startoftranscript|>`, the language
    token, the task token (`<|transcribe|>` or `<|translate|>`), and
    `<|notimestamps|>`. The same prefix is used for training and for
    `generate_preprocess()`, and it is stripped again by
    `generate_postprocess()`.

    Args:
        audio_converter: A `keras_hub.layers.WhisperAudioConverter` instance.
        tokenizer: A `keras_hub.models.WhisperTokenizer` instance.
        decoder_sequence_length: int. The maximum length of the packed decoder
            token sequence, including the control prefix. Defaults to `448`,
            matching Whisper's `max_decoder_sequence_length`.
        language: string or `None`. The language of the audio, as a language
            code such as `"en"` or `"fr"`, or as a language token such as
            `"<|fr|>"`. Only used by multilingual tokenizers. Defaults to
            `None`, which uses English on multilingual checkpoints. Automatic
            language detection is not supported.
        task: string. Either `"transcribe"` or `"translate"` (to English).
            Only used by multilingual tokenizers. Defaults to `"transcribe"`.
        return_timestamps: bool. If `False`, `<|notimestamps|>` is added to
            the prefix so the model predicts text only. If `True`, it is left
            out and the model may predict timestamp tokens, which are kept in
            the decoded text. Defaults to `False`.

    Call arguments:
        x: A dictionary with an `"audio"` key and a `"text"` key. `"audio"`
            holds raw waveforms sampled at the audio converter's sampling
            rate, of shape `(batch_size, num_samples)` or `(num_samples,)`.
            `"text"` holds the target transcriptions, as a string or a list
            or tensor of strings with one entry per waveform.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        decoder_sequence_length: Pass to override the configured
            `decoder_sequence_length` of the layer.
        sequence_length: Alias for `decoder_sequence_length`.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.WhisperAudioToTextPreprocessor.from_preset(
        "whisper_tiny_en"
    )

    # Process a single audio-text pair for training.
    inputs = {
        "audio": np.random.normal(size=(1, 16000)),
        "text": [" The quick brown fox."],
    }
    x, y, sample_weight = preprocessor(inputs)

    # Prepare inputs for generation with an optional transcript prefix.
    gen_inputs = preprocessor.generate_preprocess({
        "audio": np.random.normal(size=(1, 16000)),
        "text": [" The quick"],
    })
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.WhisperAudioToTextPreprocessor.from_preset(
        "whisper_tiny_en"
    )
    features = {
        "audio": np.random.normal(size=(2, 16000)).astype("float32"),
        "text": [" The quick brown fox.", " Jumped over the lazy dog."],
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.batch(2)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```

    Use a multilingual checkpoint to translate French speech to English.
    ```python
    preprocessor = keras_hub.models.WhisperAudioToTextPreprocessor.from_preset(
        "whisper_tiny_multi",
        language="fr",
        task="translate",
    )
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
        language=None,
        task="transcribe",
        return_timestamps=False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        if task not in TASKS:
            raise ValueError(
                f"`task` must be one of {TASKS}. Received: task={task}"
            )
        self.audio_converter = audio_converter
        self.decoder_sequence_length = decoder_sequence_length
        self.decoder_packer = None
        self._language = language
        self._task = task
        self._return_timestamps = return_timestamps

    def build(self, input_shape):
        self.decoder_packer = StartEndPacker(
            start_value=self.decoder_prefix_ids,
            end_value=self.tokenizer.eos_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @property
    def decoder_prefix_ids(self):
        """The Whisper control token ids that start every decoder sequence."""
        tokenizer = self.tokenizer
        prefix = [tokenizer.bos_token_id]
        if tokenizer.language_tokens is not None:
            language = "en" if self.language is None else self.language
            language_token = (
                language if language.startswith("<|") else f"<|{language}|>"
            )
            if language_token not in tokenizer.language_tokens:
                raise ValueError(
                    f"Unknown `language`: {language}. Expected one of "
                    f"{sorted(tokenizer.language_tokens)}."
                )
            prefix.append(tokenizer.language_tokens[language_token])
            if self.task == "translate":
                prefix.append(tokenizer.translate_token_id)
            else:
                prefix.append(tokenizer.transcribe_token_id)
        elif self.language not in (None, "en", "<|en|>") or (
            self.task != "transcribe"
        ):
            raise ValueError(
                "This is an English-only Whisper tokenizer, so `language` "
                "must be `None` or `'en'` and `task` must be `'transcribe'`. "
                f"Received: language={self.language}, task={self.task}"
            )
        if not self.return_timestamps:
            prefix.append(tokenizer.no_timestamps_token_id)
        return prefix

    def _update_prefix(self):
        if self.decoder_packer is not None:
            self.decoder_packer.start_value = self.decoder_prefix_ids

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        self._update_prefix()

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value not in TASKS:
            raise ValueError(
                f"`task` must be one of {TASKS}. Received: task={value}"
            )
        self._task = value
        self._update_prefix()

    @property
    def return_timestamps(self):
        return self._return_timestamps

    @return_timestamps.setter
    def return_timestamps(self, value):
        self._return_timestamps = value
        self._update_prefix()

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

    def _resolve_decoder_sequence_length(
        self, decoder_sequence_length, sequence_length
    ):
        return (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )

    def _generate_preprocess_python(
        self,
        x,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = self._resolve_decoder_sequence_length(
            decoder_sequence_length, sequence_length
        )
        encoder_features = self.audio_converter(x["audio"])
        # An unbatched `(num_samples,)` waveform gives unbatched
        # `(num_frames, num_mels)` features.
        unbatched = len(keras.ops.shape(encoder_features)) == 2
        if unbatched:
            encoder_features = keras.ops.expand_dims(encoder_features, 0)
        batch_size = int(keras.ops.shape(encoder_features)[0])

        decoder_text = x.get("text", None)
        if decoder_text is None:
            # The packer adds the control prefix to each empty sequence.
            decoder_token_ids = [[] for _ in range(batch_size)]
        else:
            if isinstance(decoder_text, (str, bytes)):
                decoder_text = [decoder_text]
            elif isinstance(decoder_text, np.ndarray) or (
                tf is not None and isinstance(decoder_text, tf.Tensor)
            ):
                decoder_text = np.reshape(np.asarray(decoder_text), (-1,))
                decoder_text = decoder_text.tolist()
            else:
                decoder_text = list(decoder_text)
            if len(decoder_text) == 1 and batch_size != 1:
                decoder_text = decoder_text * batch_size
            if len(decoder_text) != batch_size:
                raise ValueError(
                    "Batch size mismatch between audio "
                    f"({batch_size}) and text prompts ({len(decoder_text)})."
                )
            decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )
        outputs = {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
        if unbatched:
            outputs = keras.tree.map_structure(
                lambda t: keras.ops.squeeze(t, 0), outputs
            )
        return outputs

    @preprocessing_function
    def _generate_preprocess_tf(
        self,
        x,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = self._resolve_decoder_sequence_length(
            decoder_sequence_length, sequence_length
        )
        encoder_features = self.audio_converter(x["audio"])
        if not isinstance(encoder_features, tf.Tensor):
            encoder_features = tf.convert_to_tensor(
                keras.ops.convert_to_numpy(encoder_features)
            )
        # An unbatched `(num_samples,)` waveform gives unbatched
        # `(num_frames, num_mels)` features.
        unbatched = encoder_features.shape.rank == 2
        if unbatched:
            encoder_features = tf.expand_dims(encoder_features, 0)
        # Use `tf` ops for the batch dimension so this also works under
        # `tf.data.Dataset.map` graph tracing, where it is symbolic.
        batch_size = tf.shape(encoder_features)[0]

        decoder_text = x.get("text", None)
        if decoder_text is None:
            # The packer adds the control prefix to each empty sequence.
            decoder_token_ids = tf.RaggedTensor.from_row_lengths(
                tf.zeros([0], dtype="int32"),
                tf.zeros([batch_size], dtype="int64"),
            )
        else:
            if not isinstance(decoder_text, tf.Tensor):
                decoder_text = tf.convert_to_tensor(decoder_text)
            if decoder_text.shape.rank == 0:
                # Broadcast a scalar prompt to every audio in the batch.
                decoder_text = tf.fill([batch_size], decoder_text)
            else:
                text_batch_size = decoder_text.shape[0]
                audio_batch_size = encoder_features.shape[0]
                if text_batch_size == 1 and audio_batch_size != 1:
                    decoder_text = tf.repeat(decoder_text, batch_size, axis=0)
                elif (
                    text_batch_size is not None
                    and audio_batch_size is not None
                    and text_batch_size != audio_batch_size
                ):
                    raise ValueError(
                        "Batch size mismatch between audio "
                        f"({audio_batch_size}) and text prompts "
                        f"({text_batch_size})."
                    )
            decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )
        outputs = {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
        if unbatched:
            outputs = {k: tf.squeeze(v, 0) for k, v in outputs.items()}
        return outputs

    def generate_preprocess(
        self,
        x,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        """Convert raw audio (and an optional text prompt) to model inputs.

        This mirrors `call()`, but does not append the end token to the decoder
        sequence and does not produce labels. Every sequence starts with the
        control prefix (see `decoder_prefix_ids`). If `x` has a `"text"` key,
        its tokens follow the prefix and generation continues from them. A
        single prompt string is broadcast to every waveform in the batch.
        """
        if self._use_tf_workflow():
            return self._generate_preprocess_tf(
                x,
                decoder_sequence_length=decoder_sequence_length,
                sequence_length=sequence_length,
            )
        return self._generate_preprocess_python(
            x,
            decoder_sequence_length=decoder_sequence_length,
            sequence_length=sequence_length,
        )

    def _ids_to_strip(self):
        ids_to_strip = list(self.tokenizer.special_token_ids)
        if self.tokenizer.language_tokens is not None:
            ids_to_strip += list(self.tokenizer.language_tokens.values())
        return ids_to_strip

    def _generate_postprocess_python(self, x):
        if not self.built:
            self.build(None)
        token_ids = strip_to_ragged_python(
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
            self._ids_to_strip(),
        )
        return self.tokenizer.detokenize(token_ids)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        if not self.built:
            self.build(None)
        token_ids = keras.ops.convert_to_numpy(x["decoder_token_ids"])
        padding_mask = keras.ops.convert_to_numpy(x["decoder_padding_mask"])
        token_ids = strip_to_ragged(
            token_ids, padding_mask, self._ids_to_strip()
        )
        return self.tokenizer.detokenize(token_ids)

    def generate_postprocess(self, x):
        """Convert generated token ids back to strings.

        Padding, the control prefix, language tokens, and other special tokens
        are removed before detokenizing.
        """
        if self._use_tf_workflow():
            return self._generate_postprocess_tf(x)
        return self._generate_postprocess_python(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_sequence_length": self.decoder_sequence_length,
                "language": self.language,
                "task": self.task,
                "return_timestamps": self.return_timestamps,
            }
        )
        return config
