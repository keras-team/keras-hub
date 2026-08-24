import math

import numpy as np
from keras import KerasTensor
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone


@keras_hub_export("keras_hub.layers.Gemma3nAudioConverter")
class Gemma3nAudioConverter(AudioConverter):
    """Converts raw audio waveforms into log-mel spectrograms.

    This layer preprocesses 1D audio signals into 2D log-mel spectrograms
    suitable for the Gemma3n audio encoder. The conversion process involves
    padding or truncating the raw audio to a consistent length, applying
    optional dithering, input scaling, and preemphasis, and then computing the
    Short-Time Fourier Transform (STFT) with a Hann window. The resulting
    magnitude spectrogram is converted to the mel scale using a mel filterbank,
    after which the log-mel spectrogram is calculated by taking the logarithm.
    Finally, the layer can optionally normalize these features using provided
    per-bin mean and standard deviation statistics, and it returns both the
    spectrogram and an attention mask indicating which frames are valid.

    Args:
        feature_size: int. The number of mel bins to generate.
        sampling_rate: int. The expected sampling rate of the input audio.
        padding_value: float. The value to use for padding the raw audio.
        return_attention_mask: bool. Whether to return an attention mask.
        frame_length_ms: float. The length of each STFT frame in
            milliseconds.
        hop_length_ms: float. The step size between STFT frames in
            milliseconds.
        min_frequency: float. The lowest frequency for the mel filterbank.
        max_frequency: float. The highest frequency for the mel filterbank.
        preemphasis: float. The coefficient for the preemphasis filter.
            Set to 0.0 to disable.
        preemphasis_htk_flavor: bool. Whether to use the HTK-style
            preemphasis.
        fft_overdrive: bool. If True, doubles the FFT length.
        dither: float. Amount of dithering to add to the waveform.
            Set to 0.0 to disable.
        input_scale_factor: float. Factor to scale the input waveform by.
        mel_floor: float. A minimum value (floor) to apply before taking
            the logarithm.
        per_bin_mean: list or None. A list of mean values for each mel
            bin, used for normalization.
        per_bin_stddev: list or None. A list of standard deviation values
            for each mel bin, used for normalization.
        padding_side: str. Which side to pad the audio on ('right' or
            'left').

    Call arguments:
        raw_speech: A raw audio waveform tensor, list of waveforms, or numpy
            array. Can be unbatched (1D) or batched (list of 1D arrays).
        padding: str or bool. Padding strategy for batches. Options are
            `"longest"` (pad to longest sequence in batch), `True` (same as
            `"longest"`), or `False` (no padding). Defaults to `"longest"`.
        max_length: int. Maximum length to truncate or pad to. Defaults to
            480000.
        truncation: bool. Whether to truncate sequences longer than
            `max_length`. Defaults to `True`.
        pad_to_multiple_of: int or None. If set, pad the sequence length to a
            multiple of this value. Defaults to 128.
        return_attention_mask: bool. Whether to return an attention mask
            indicating valid (non-padded) frames. Defaults to `True`.

    Examples:
    ```python
    import numpy as np

    audio = np.sin(
        2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32)
    )

    converter = keras_hub.layers.Gemma3nAudioConverter(
        feature_size=128,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=True,
        frame_length_ms=32.0,
        hop_length_ms=10.0,
        min_frequency=125.0,
        max_frequency=7600.0,
        preemphasis=0.97,
        preemphasis_htk_flavor=True,
        fft_overdrive=True,
        dither=0.0,
        input_scale_factor=1.0,
        mel_floor=1e-5,
        per_bin_mean=None,
        per_bin_stddev=None,
        padding_side="right",
    )

    features, mask = converter(audio)
    print(features.shape)
    print(mask.shape)
    ```
    """

    backbone_cls = Gemma3nBackbone

    def __init__(
        self,
        feature_size,
        sampling_rate,
        padding_value,
        return_attention_mask,
        frame_length_ms,
        hop_length_ms,
        min_frequency,
        max_frequency,
        preemphasis,
        preemphasis_htk_flavor,
        fft_overdrive,
        dither,
        input_scale_factor,
        mel_floor,
        per_bin_mean,
        per_bin_stddev,
        padding_side,
        **kwargs,
    ):
        # === Config ===
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        self.padding_side = padding_side
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.mel_floor = mel_floor
        self.per_bin_mean = per_bin_mean
        self.per_bin_stddev = per_bin_stddev
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length
        hann_arange = np.arange(self.frame_length, dtype=np.float32)
        self.window = ops.convert_to_tensor(
            0.5 * (1.0 - np.cos(2.0 * np.pi * hann_arange / self.frame_length)),
            dtype=self.compute_dtype,
        )
        self.mel_filters = self._create_filterbank_matrix(
            n_freqs=self.fft_length // 2 + 1,
            f_min=min_frequency,
            f_max=max_frequency,
            n_mels=feature_size,
            sample_rate=self.sampling_rate,
            fft_length=fft_length,
        )

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.built = True

    def _create_filterbank_matrix(
        self,
        n_freqs,
        f_min,
        f_max,
        n_mels,
        sample_rate,
        fft_length,
    ):
        # Construct the filterbank in float64 with NumPy at initialization
        # time for numerical consistency with the reference implementation.
        all_freqs = np.arange(n_freqs, dtype=np.float64) * (
            sample_rate / fft_length
        )
        # HTK mel-scale formula:
        #   mel  = 2595 * log10(1 + f / 700)
        #   freq = 700 * (10^(mel / 2595) - 1)
        # The constants 2595.0 and 700.0 define the linear-to-logarithmic
        # crossover that models human auditory perception. They are the
        # standard values used by HTK, Kaldi, and librosa.
        # Ref: https://en.wikipedia.org/wiki/Mel_scale#Formula
        m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
        m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
        m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
        f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[None, :] - all_freqs[:, None]
        down_slopes = -slopes[:, :-2] / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
        return ops.convert_to_tensor(fb, dtype=self.compute_dtype)

    def _extract_spectrogram(self, waveform, attention_mask=None):
        waveform = ops.cast(
            waveform,
            dtype=self.compute_dtype,
        )
        waveform_rank = ops.ndim(waveform)
        if waveform_rank not in (1, 2):
            raise ValueError(
                "`waveform` must have rank 1 or 2. "
                f"Received rank={waveform_rank}, "
                f"shape={ops.shape(waveform)}."
            )
        if self.dither > 0.0:
            waveform = waveform + self.dither * random.normal(
                ops.shape(waveform),
                dtype=waveform.dtype,
            )
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor
        frames_to_process = ops.extract_sequences(
            waveform,
            sequence_length=self.frame_length + 1,
            sequence_stride=self.hop_length,
        )
        # Pre-emphasis
        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_sample = frames_to_process[..., :1] * (
                    1.0 - self.preemphasis
                )

                rest_of_samples = (
                    frames_to_process[..., 1:-1]
                    - self.preemphasis * frames_to_process[..., :-2]
                )
                frames = ops.concatenate(
                    [first_sample, rest_of_samples], axis=-1
                )
            else:
                frames = (
                    frames_to_process[..., 1:]
                    - self.preemphasis * frames_to_process[..., :-1]
                )
        else:
            frames = frames_to_process[..., :-1]
        frames = frames * self.window
        fft_pad = self.fft_length - self.frame_length
        if fft_pad > 0:
            if waveform_rank == 1:
                frames = ops.pad(frames, [[0, 0], [0, fft_pad]])
            else:
                frames = ops.pad(frames, [[0, 0], [0, 0], [0, fft_pad]])

        stft = ops.rfft(frames, fft_length=self.fft_length)
        if isinstance(stft, (tuple, list)):
            real, imag = stft
            magnitude_spec = ops.sqrt(ops.square(real) + ops.square(imag))
        else:
            magnitude_spec = ops.abs(stft)
        mel_spec = ops.matmul(magnitude_spec, self.mel_filters)
        mel_floor_tensor = ops.cast(self.mel_floor, dtype=self.compute_dtype)
        log_mel_spec = ops.log(ops.maximum(mel_spec, mel_floor_tensor))
        if self.per_bin_mean is not None:
            mean = ops.convert_to_tensor(
                self.per_bin_mean, dtype=self.compute_dtype
            )
            mean = ops.reshape(mean, (1, self.feature_size))
            log_mel_spec = log_mel_spec - mean
        if self.per_bin_stddev is not None:
            stddev = ops.convert_to_tensor(
                self.per_bin_stddev, dtype=self.compute_dtype
            )
            stddev = ops.reshape(stddev, (1, self.feature_size))
            log_mel_spec = log_mel_spec / stddev
        mel_spectrogram = ops.cast(log_mel_spec, dtype=self.compute_dtype)

        num_output_frames = ops.shape(mel_spectrogram)[-2]
        if attention_mask is None:
            mask = None
        else:
            frame_masks = ops.extract_sequences(
                attention_mask,
                sequence_length=self.frame_length,
                sequence_stride=self.hop_length,
            )
            frame_masks = ops.cast(frame_masks, dtype="bool")
            mask = ops.all(frame_masks, axis=-1)
            mask = mask[..., :num_output_frames]
        return mel_spectrogram, mask

    def _get_padding_strategies(self, padding=False, max_length=None):
        if padding is not False:
            if padding is True:
                padding_strategy = "longest"
            else:
                padding_strategy = padding
        else:
            padding_strategy = "do_not_pad"

        if max_length is None:
            if padding_strategy == "max_length":
                raise ValueError(
                    "When setting padding='max_length', max_length must "
                    "be defined"
                )

        if padding_strategy != "do_not_pad" and self.padding_value is None:
            raise ValueError("Padding requested but no padding_value defined")
        return padding_strategy

    def _pad(
        self,
        input_features,
        attention_mask=None,
        max_length=None,
        padding_strategy="do_not_pad",
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ):
        required_input = input_features

        if padding_strategy == "longest":
            max_length = len(required_input)
        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and max_length % pad_to_multiple_of != 0
        ):
            max_length = (
                (max_length // pad_to_multiple_of) + 1
            ) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != "do_not_pad"
            and len(required_input) < max_length
        )
        if return_attention_mask and attention_mask is None:
            attention_mask = np.ones(len(required_input), dtype=np.int32)
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    attention_mask = np.pad(
                        attention_mask,
                        (0, difference),
                    )
                if required_input.ndim > 1:
                    padding_shape = ((0, difference), (0, 0))
                else:
                    padding_shape = ((0, difference),)
                input_features = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    attention_mask = np.pad(attention_mask, (difference, 0))
                if required_input.ndim > 1:
                    padding_shape = ((difference, 0), (0, 0))
                else:
                    padding_shape = (difference, 0)
                input_features = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
        return input_features, attention_mask

    def _truncate(
        self,
        input_features,
        attention_mask=None,
        max_length=None,
        pad_to_multiple_of=None,
        truncation=None,
    ):
        if not truncation:
            return input_features, attention_mask

        if truncation and max_length is None:
            raise ValueError(
                "When setting truncation=True, max_length must be defined"
            )
        required_input = input_features
        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and max_length % pad_to_multiple_of != 0
        ):
            max_length = (
                (max_length // pad_to_multiple_of) + 1
            ) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length
        if needs_to_be_truncated:
            input_features = input_features[:max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:max_length]
        return input_features, attention_mask

    def pad(
        self,
        input_features,
        padding=True,
        max_length=None,
        truncation=False,
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ):
        required_input = input_features
        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )
        if len(required_input) == 0:
            if return_attention_mask:
                return [], []
            return [], None
        required_input = [np.asarray(v) for v in required_input]
        padding_strategy = self._get_padding_strategies(
            padding=padding, max_length=max_length
        )
        batch_size = len(required_input)
        truncated_inputs = []
        truncated_masks = []
        for i in range(batch_size):
            inputs = required_input[i]
            mask = (
                np.ones(len(inputs), dtype=np.int32)
                if return_attention_mask
                else None
            )
            inputs_slice, mask_slice = self._truncate(
                inputs,
                attention_mask=mask,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)
            if mask_slice is not None:
                truncated_masks.append(mask_slice)
        if padding_strategy == "longest":
            max_length = max(
                len(input_slice) for input_slice in truncated_inputs
            )
            padding_strategy = "max_length"
        batch_outputs_features = []
        batch_outputs_masks = []
        for i in range(batch_size):
            inputs = truncated_inputs[i]
            mask = truncated_masks[i] if return_attention_mask else None
            outputs_features, outputs_mask = self._pad(
                inputs,
                attention_mask=mask,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            if outputs_features.dtype == np.dtype(np.float64):
                outputs_features = outputs_features.astype(np.float32)
            batch_outputs_features.append(outputs_features)
            if outputs_mask is not None:
                batch_outputs_masks.append(outputs_mask)
        if not return_attention_mask:
            return batch_outputs_features, None
        return batch_outputs_features, batch_outputs_masks

    def compute_output_spec(
        self, raw_speech, return_attention_mask=None, *args, **kwargs
    ):
        was_batched = len(raw_speech.shape) > 1
        if was_batched:
            features_shape = (
                raw_speech.shape[0],
                None,
                self.feature_size,
            )
            mask_shape = (raw_speech.shape[0], None)
        else:
            features_shape = (None, self.feature_size)
            mask_shape = (None,)

        features_spec = KerasTensor(
            shape=features_shape,
            dtype=self.compute_dtype,
        )
        if return_attention_mask is None:
            return_attention_mask = self.return_attention_mask

        if return_attention_mask:
            return (
                features_spec,
                KerasTensor(shape=mask_shape, dtype="int32"),
            )
        return features_spec, None

    def call(
        self,
        raw_speech,
        padding="longest",
        max_length=480000,
        truncation=True,
        pad_to_multiple_of=128,
        return_attention_mask=True,
    ):
        import tensorflow as tf

        if isinstance(raw_speech, KerasTensor):
            return self.compute_output_spec(raw_speech)
        if isinstance(raw_speech, (list, tuple)):
            speech_list = [
                np.asarray(speech).reshape(-1) for speech in raw_speech
            ]
            input_features_list, attention_mask_list = self.pad(
                speech_list,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            if not return_attention_mask:
                attention_mask_list = [None] * len(input_features_list)
            prepared_features = []
            prepared_masks = []

            for speech, mask in zip(input_features_list, attention_mask_list):
                speech = ops.convert_to_tensor(
                    np.asarray(speech).reshape(-1), dtype=self.compute_dtype
                )

                mask_tensor = (
                    ops.convert_to_tensor(
                        np.asarray(mask).reshape(-1), dtype="int32"
                    )
                    if mask is not None
                    else None
                )

                # Single waveform -> 1-D input.
                features, feature_mask = self._extract_spectrogram(
                    speech, mask_tensor
                )
                features = ops.reshape(features, (-1, self.feature_size))
                if return_attention_mask:
                    if feature_mask is None:
                        feature_mask = ops.ones(
                            (ops.shape(features)[0],), dtype="int32"
                        )
                    else:
                        feature_mask = ops.reshape(feature_mask, (-1,))
                        feature_mask = ops.cast(feature_mask, "int32")

                    prepared_masks.append(feature_mask)
                prepared_features.append(features)

            input_features = ops.stack(prepared_features, axis=0)
            if return_attention_mask:
                input_features_mask = ops.stack(prepared_masks, axis=0)
            else:
                input_features_mask = None
            return input_features, input_features_mask

        if isinstance(raw_speech, (list, tuple)):
            raw_speech_tensor = ops.stack(
                [
                    speech
                    if ops.is_tensor(speech)
                    else ops.convert_to_tensor(speech)
                    for speech in raw_speech
                ],
                axis=0,
            )
        elif ops.is_tensor(raw_speech):
            raw_speech_tensor = raw_speech
        else:
            raw_speech_tensor = ops.convert_to_tensor(raw_speech)

        rank = ops.ndim(raw_speech_tensor)
        if rank == 1:
            speech_np = np.asarray(raw_speech_tensor).reshape(-1)

            input_features_list, attention_mask_list = self.pad(
                [speech_np],
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            if not input_features_list:
                features = ops.zeros(
                    (0, self.feature_size), dtype=self.compute_dtype
                )
                mask = (
                    ops.zeros((0,), dtype="int32")
                    if return_attention_mask
                    else None
                )
                return features, mask

            speech = ops.convert_to_tensor(
                input_features_list[0], dtype=self.compute_dtype
            )
            if return_attention_mask:
                mask = ops.convert_to_tensor(
                    attention_mask_list[0], dtype="int32"
                )
            else:
                mask = None
            features, feature_mask = self._extract_spectrogram(speech, mask)
            features = ops.reshape(features, (-1, self.feature_size))

            if return_attention_mask:
                if feature_mask is None:
                    feature_mask = ops.ones(
                        (ops.shape(features)[0],), dtype="int32"
                    )
                else:
                    feature_mask = ops.reshape(feature_mask, (-1,))
                    feature_mask = ops.cast(feature_mask, "int32")
            else:
                feature_mask = None
            return features, feature_mask

        def process_one_audio(speech):
            speech = ops.reshape(speech, (-1,))
            speech = ops.cast(speech, self.compute_dtype)
            speech_batched = ops.expand_dims(speech, axis=0)
            speech_length = ops.shape(speech_batched)[1]
            speech_mask = ops.ones((1, speech_length), dtype="int32")
            features, feature_mask = self._extract_spectrogram(
                speech_batched, speech_mask
            )
            features = ops.reshape(features, (-1, self.feature_size))
            features = ops.cast(features, self.compute_dtype)
            if feature_mask is None:
                num_frames = ops.shape(features)[0]
                feature_mask = ops.ones((num_frames,), dtype="int32")
            else:
                feature_mask = ops.reshape(feature_mask, (-1,))
                feature_mask = ops.cast(feature_mask, "int32")
            return features, feature_mask

        input_features, input_features_mask = tf.map_fn(
            process_one_audio,
            raw_speech_tensor,
            fn_output_signature=(
                tf.TensorSpec(
                    shape=(None, self.feature_size),
                    dtype=tf.as_dtype(self.compute_dtype),
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
        )
        return input_features, input_features_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feature_size": self.feature_size,
                "sampling_rate": self.sampling_rate,
                "padding_value": self.padding_value,
                "return_attention_mask": self.return_attention_mask,
                "frame_length_ms": self.frame_length_ms,
                "hop_length_ms": self.hop_length_ms,
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "preemphasis": self.preemphasis,
                "preemphasis_htk_flavor": self.preemphasis_htk_flavor,
                "fft_overdrive": self.fft_overdrive,
                "dither": self.dither,
                "input_scale_factor": self.input_scale_factor,
                "mel_floor": self.mel_floor,
                "per_bin_mean": self.per_bin_mean,
                "per_bin_stddev": self.per_bin_stddev,
                "padding_side": self.padding_side,
            }
        )

        return config
