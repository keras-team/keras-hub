import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.layers.Qwen3ASRAudioConverter")
class Qwen3ASRAudioConverter(WhisperAudioConverter):
    """Qwen3 ASR audio converter layer."""

    backbone_cls = Qwen3ASRBackbone

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
        n_window=50,
        min_length=8000,
        **kwargs,
    ):
        super().__init__(
            num_mels=num_mels,
            num_fft_bins=num_fft_bins,
            stride=stride,
            sampling_rate=sampling_rate,
            max_audio_length=max_audio_length,
            **kwargs,
        )
        self.n_window = n_window
        self.min_length = min_length

    def audio_shape(self):
        return (None, self.num_mels)

    def compute_output_spec(self, audio):
        if len(audio.shape) == 1:
            sample_length = audio.shape[0]
            batch_shape = None
        else:
            sample_length = audio.shape[1]
            batch_shape = audio.shape[0]
        if sample_length is not None:
            sample_length = max(sample_length, self.min_length)
            frame_length = sample_length // self.stride
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

    def _to_batched_array(self, audio):
        """Convert audio to a batch padded to its longest clip."""
        if tf is not None and isinstance(audio, tf.RaggedTensor):
            rows, unbatched = audio.to_list(), False
        else:
            try:
                array = convert_to_numpy(audio)
            except ValueError:
                array = None
            if array is None or array.dtype == object:
                rows, unbatched = list(audio), False
            elif array.ndim == 1:
                rows, unbatched = [array], True
            else:
                rows, unbatched = list(array), False

        rows = [np.reshape(convert_to_numpy(row), (-1,)) for row in rows]
        max_length = max(
            self.min_length,
            max((row.shape[0] for row in rows), default=0),
        )
        batch = np.zeros((len(rows), max_length), dtype=self.compute_dtype)
        for i, row in enumerate(rows):
            batch[i, : row.shape[0]] = row
        return batch, unbatched

    @preprocessing_function
    def _call_tf(self, audio):
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            try:
                audio = tf.convert_to_tensor(audio)
            except ValueError:
                audio = tf.ragged.constant(audio)

        rank_1_input = audio.shape.rank == 1
        if rank_1_input:
            audio = tf.expand_dims(audio, 0)

        if isinstance(audio, tf.RaggedTensor):
            audio = audio.to_tensor()

        curr_len = tf.shape(audio)[1]
        target_len = tf.maximum(curr_len, self.min_length)
        audio = tf.pad(audio, [[0, 0], [0, target_len - curr_len]])

        log_spec = self._extract_audio_features(audio)
        if rank_1_input:
            log_spec = tf.squeeze(log_spec, 0)
        return log_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_window": self.n_window,
                "min_length": self.min_length,
            }
        )
        return config
