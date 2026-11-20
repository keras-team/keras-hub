import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def _to_host(x):
    if tf is not None and isinstance(x, (tf.Tensor, tf.RaggedTensor)):
        return x
    if isinstance(x, list):
        return [_to_host(item) for item in x]
    if isinstance(x, tuple):
        return tuple(_to_host(item) for item in x)
    if keras.ops.is_tensor(x):
        if keras.config.backend() == "torch":
            return x.cpu()
        return convert_to_numpy(x)
    return x


@keras_hub_export("keras_hub.layers.Qwen3ASRAudioConverter")
class Qwen3ASRAudioConverter(WhisperAudioConverter):
    """Qwen3 ASR audio converter layer.

    Converts raw waveform audio into log-mel spectrograms for the Qwen3-ASR
    model. Short clips are padded to `min_length` before feature extraction so
    the encoder sees a consistent chunk length.

    Args:
        num_mels: int. Number of mel-frequency bins. Defaults to `128`.
        num_fft_bins: int. Size of the Fourier transform. Defaults to `400`.
        stride: int. Hop length in samples between STFT frames. Defaults to
            `160`.
        sampling_rate: int. Audio sampling rate in Hz. Defaults to `16000`.
        n_window: int. Audio chunk size half-window for token counting.
            Defaults to `50`.
        min_length: int. Minimum number of samples to pad to before feature
            extraction. Defaults to `8000`.

    Example:
    ```python
    converter = keras_hub.layers.Qwen3ASRAudioConverter()
    audio = np.ones((16000,), dtype="float32")
    converter(audio)
    ```
    """

    backbone_cls = Qwen3ASRBackbone

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        n_window=50,
        min_length=8000,
        **kwargs,
    ):
        super().__init__(
            num_mels=num_mels,
            num_fft_bins=num_fft_bins,
            stride=stride,
            sampling_rate=sampling_rate,
            max_audio_length=30,
            **kwargs,
        )
        self.n_window = n_window
        self.min_length = min_length

    def audio_shape(self):
        return (None, self.num_mels)

    def compute_output_spec(self, audio):
        if len(audio.shape) not in (1, 2):
            raise ValueError(
                f"Expected 1D or 2D audio input, got shape {audio.shape}."
            )
        if len(audio.shape) == 1:
            sample_length = audio.shape[0]
            batch_shape = None
        else:
            sample_length = audio.shape[1]
            batch_shape = audio.shape[0]
        if sample_length is not None:
            sample_length = max(sample_length, self.min_length)
            frame_length = (
                sample_length + 2 * (self.num_fft_bins // 2) - self.num_fft_bins
            ) // self.stride
        else:
            frame_length = None
        if len(audio.shape) == 1:
            output_shape = (frame_length, self.num_mels)
        else:
            output_shape = (batch_shape, frame_length, self.num_mels)
        return keras.KerasTensor(
            shape=output_shape,
            dtype=self.compute_dtype,
        )

    def _extract_rows(self, audio):
        """Convert audio input into a list of 1D NumPy arrays."""
        if tf is not None and isinstance(audio, tf.RaggedTensor):
            if audio.shape.rank is not None and audio.shape.rank > 2:
                raise ValueError(
                    "Expected 1D or 2D audio input, got rank "
                    f"{audio.shape.rank}."
                )
            rows, unbatched = audio.to_list(), False
        else:
            try:
                array = convert_to_numpy(audio)
            except (ValueError, TypeError):
                array = None
            if array is None or array.dtype == object:
                rows, unbatched = list(audio), False
            elif array.ndim == 1:
                rows, unbatched = [array], True
            elif array.ndim == 2:
                rows, unbatched = list(array), False
            else:
                raise ValueError(
                    f"Expected 1D or 2D audio input, got shape {array.shape}."
                )

        parsed_rows = []
        for row in rows:
            row_arr = convert_to_numpy(row)
            if row_arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D audio clip, got shape {row_arr.shape}."
                )
            parsed_rows.append(row_arr.astype(self.compute_dtype, copy=False))
        return parsed_rows, unbatched

    def _to_batched_array(self, audio):
        """Convert audio to a batch padded to its longest clip."""
        rows, unbatched = self._extract_rows(audio)
        max_length = max(
            self.min_length,
            max((row.shape[0] for row in rows), default=0),
        )
        batch = np.zeros((len(rows), max_length), dtype=self.compute_dtype)
        for i, row in enumerate(rows):
            batch[i, : row.shape[0]] = row
        return batch, unbatched

    def _call_python(self, audio):
        rows, rank_1_input = self._extract_rows(audio)
        if not rows:
            log_spec = np.zeros((0, 0, self.num_mels), dtype=self.compute_dtype)
        elif (
            len({row.shape[0] for row in rows}) == 1
            and rows[0].shape[0] >= self.min_length
        ):
            batch = np.stack(rows, axis=0)
            log_spec = self._extract_audio_features_python(batch)
        else:
            clip_mels = []
            max_frames = 0
            for row in rows:
                if row.shape[0] < self.min_length:
                    padded_row = np.zeros(
                        (1, self.min_length), dtype=self.compute_dtype
                    )
                    padded_row[0, : row.shape[0]] = row
                else:
                    padded_row = row[np.newaxis, :]
                clip_mel = self._extract_audio_features_python(padded_row)[0]
                clip_mels.append(clip_mel)
                max_frames = max(max_frames, clip_mel.shape[0])
            log_spec = np.zeros(
                (len(rows), max_frames, self.num_mels),
                dtype=self.compute_dtype,
            )
            for i, clip_mel in enumerate(clip_mels):
                log_spec[i, : clip_mel.shape[0], :] = clip_mel
        if rank_1_input:
            log_spec = np.squeeze(log_spec, 0)
        return convert_preprocessing_outputs_python(log_spec)

    @preprocessing_function
    def _call_tf(self, audio):
        audio = _to_host(audio)
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            try:
                audio = tf.convert_to_tensor(audio)
            except ValueError:
                audio = tf.ragged.constant(audio)

        rank = audio.shape.rank
        if rank is not None and rank not in (1, 2):
            raise ValueError(f"Expected 1D or 2D audio input, got rank {rank}.")

        rank_1_input = rank == 1
        if rank_1_input:
            audio = tf.expand_dims(audio, 0)

        if isinstance(audio, tf.RaggedTensor):

            def extract_single_clip(row):
                curr_len = tf.shape(row)[0]
                target_len = tf.maximum(curr_len, self.min_length)
                row = tf.pad(row, [[0, target_len - curr_len]])
                return self._extract_audio_features(tf.expand_dims(row, 0))[0]

            log_spec = tf.map_fn(
                extract_single_clip,
                audio,
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[None, self.num_mels],
                    ragged_rank=0,
                    dtype=self.compute_dtype,
                ),
            )
            log_spec = log_spec.to_tensor(default_value=0.0)
        else:
            curr_len = tf.shape(audio)[1]
            target_len = tf.maximum(curr_len, self.min_length)
            audio = tf.pad(audio, [[0, 0], [0, target_len - curr_len]])
            log_spec = self._extract_audio_features(audio)

        if rank_1_input:
            log_spec = tf.squeeze(log_spec, 0)
        return log_spec

    def get_config(self):
        config = super().get_config()
        config.pop("max_audio_length", None)
        config.update(
            {
                "n_window": self.n_window,
                "min_length": self.min_length,
            }
        )
        return config
