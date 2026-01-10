import math

import keras
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None
from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.Gemma3nAudioConverter")
class Gemma3nAudioConverter(keras.layers.Layer):
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
            Defaults to 128.
        sampling_rate: int. The expected sampling rate of the input audio.
            Defaults to 16000.
        padding_value: float. The value to use for padding the raw audio.
            Defaults to 0.0.
        return_attention_mask: bool. Whether to return an attention mask.
            Defaults to True.
        frame_length_ms: float. The length of each STFT frame in
            milliseconds. Defaults to 32.0.
        hop_length_ms: float. The step size between STFT frames in
            milliseconds. Defaults to 10.0.
        min_frequency: float. The lowest frequency for the mel filterbank.
            Defaults to 125.0.
        max_frequency: float. The highest frequency for the mel filterbank.
            Defaults to 7600.0.
        preemphasis: float. The coefficient for the preemphasis filter.
            Set to 0.0 to disable. Defaults to 0.97.
        preemphasis_htk_flavor: bool. Whether to use the HTK-style
            preemphasis. Defaults to True.
        fft_overdrive: bool. If True, doubles the FFT length.
            Defaults to True.
        dither: float. Amount of dithering to add to the waveform.
            Set to 0.0 to disable. Defaults to 0.0.
        input_scale_factor: float. Factor to scale the input waveform by.
            Defaults to 1.0.
        mel_floor: float. A minimum value (floor) to apply before taking
            the logarithm. Defaults to 1e-5.
        per_bin_mean: list or None. A list of mean values for each mel
            bin, used for normalization. Defaults to None.
        per_bin_stddev: list or None. A list of standard deviation values
            for each mel bin, used for normalization. Defaults to None.
        padding_side: str. Which side to pad the audio on ('right' or
            'left'). Defaults to 'right'.
    """

    def __init__(
        self,
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
        self.mel_floor_arg = mel_floor
        self.per_bin_mean_arg = per_bin_mean
        self.per_bin_stddev_arg = per_bin_stddev
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        self.mel_floor = tf.constant(mel_floor, dtype=self.compute_dtype)
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length
        hann_arange = tf.range(self.frame_length, dtype=self.compute_dtype)
        self.window = 0.5 * (
            1 - tf.cos(2 * np.pi * hann_arange / self.frame_length)
        )
        self.mel_filters = self._create_fb_matrix(
            n_freqs=self.fft_length // 2 + 1,
            f_min=min_frequency,
            f_max=max_frequency,
            n_mels=feature_size,
            sample_rate=self.sampling_rate,
            fft_length=fft_length,
        )
        if per_bin_mean is not None:
            self.per_bin_mean = tf.constant(
                per_bin_mean,
                shape=(1, 1, feature_size),
                dtype=self.compute_dtype,
            )
        else:
            self.per_bin_mean = None
        if per_bin_stddev is not None:
            self.per_bin_stddev = tf.constant(
                per_bin_stddev,
                shape=(1, 1, feature_size),
                dtype=self.compute_dtype,
            )
        else:
            self.per_bin_stddev = None
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.built = True

    def _create_fb_matrix(
        self,
        n_freqs,
        f_min,
        f_max,
        n_mels,
        sample_rate,
        fft_length,
    ):
        all_freqs = tf.cast(tf.range(n_freqs), dtype=self.compute_dtype) * (
            sample_rate / fft_length
        )
        m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        f_pts = tf.constant(f_pts, dtype=self.compute_dtype)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = tf.expand_dims(f_pts, 0) - tf.expand_dims(all_freqs, 1)
        zero = tf.zeros(1, dtype=self.compute_dtype)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = tf.maximum(zero, tf.minimum(down_slopes, up_slopes))
        return tf.constant(fb, dtype=self.compute_dtype)

    def _extract_spectrogram(self, waveform, attention_mask):
        waveform = tf.cast(waveform, dtype=self.compute_dtype)
        if self.dither > 0.0:
            waveform = waveform + self.dither * tf.random.normal(
                tf.shape(waveform), dtype=waveform.dtype
            )
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor
        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_sample = waveform[:, :1] * (1.0 - self.preemphasis)
                rest_of_samples = (
                    waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
                )
                waveform = tf.concat([first_sample, rest_of_samples], axis=-1)
            else:
                waveform = tf.concat(
                    [
                        waveform[:, :1],
                        waveform[:, 1:] - self.preemphasis * waveform[:, :-1],
                    ],
                    axis=-1,
                )
        frames = tf.signal.frame(
            waveform,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            pad_end=False,
        )
        frames = frames * self.window
        pad_length = self.fft_length - self.frame_length
        paddings = [[0, 0], [0, 0], [0, pad_length]]
        frames = tf.pad(frames, paddings)
        stft = tf.signal.rfft(frames)
        magnitude_spec = tf.abs(stft)
        mel_spec = tf.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = tf.math.log(tf.maximum(mel_spec, self.mel_floor))
        if self.per_bin_mean is not None:
            log_mel_spec = log_mel_spec - self.per_bin_mean
        if self.per_bin_stddev is not None:
            log_mel_spec = log_mel_spec / self.per_bin_stddev
        mel_spectrogram = tf.squeeze(log_mel_spec, axis=0)
        mask = tf.cast(attention_mask[:: self.hop_length], dtype=tf.bool)
        return mel_spectrogram, mask[: tf.shape(mel_spectrogram)[0]]

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
                    "When setting padding='max_length', max_length must be "
                    "defined"
                )
        if padding_strategy != "do_not_pad" and (self.padding_value is None):
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
            and (max_length % pad_to_multiple_of != 0)
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
                    attention_mask = np.pad(attention_mask, (0, difference))
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
                    padding_shape = ((difference, 0),)
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
        elif truncation and max_length is None:
            raise ValueError(
                "When setting truncation=True, max_length must be defined"
            )
        required_input = input_features
        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
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
            return [], [] if return_attention_mask else None
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

    def call(
        self,
        raw_speech,
        padding="longest",
        max_length=480000,
        truncation=True,
        pad_to_multiple_of=128,
        return_attention_mask=True,
    ):
        def _process_in_py(raw_speech_tensor):
            raw_speech_np = raw_speech_tensor.numpy()
            is_batched = raw_speech_np.ndim > 1
            if is_batched:
                speech_list = [rs.reshape(-1, 1) for rs in raw_speech_np]
            else:
                raw_speech_np = np.atleast_1d(raw_speech_np)
                speech_list = [raw_speech_np.reshape(-1, 1)]
            input_features_list, attention_mask_list = self.pad(
                speech_list,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            prepared_speech = []
            prepared_speech_mask = []
            for speech, mask in zip(input_features_list, attention_mask_list):
                speech_tensor = tf.constant(speech.T, dtype=self.compute_dtype)
                mask_tensor = tf.constant(mask, dtype=tf.int32)
                features, feature_mask = self._extract_spectrogram(
                    speech_tensor, mask_tensor
                )
                prepared_speech.append(features)
                prepared_speech_mask.append(feature_mask)
            input_features = tf.stack(prepared_speech)
            input_features_mask = tf.stack(prepared_speech_mask)
            if not is_batched:
                input_features = tf.squeeze(input_features, axis=0)
                input_features_mask = tf.squeeze(input_features_mask, axis=0)
            return input_features, input_features_mask

        if not isinstance(raw_speech, (tf.Tensor, tf.RaggedTensor)):
            was_batched = isinstance(raw_speech, (list, tuple))
            raw_speech = tf.convert_to_tensor(
                raw_speech, dtype=self.compute_dtype
            )
        else:
            was_batched = raw_speech.shape.rank > 1
        input_features, input_features_mask = tf.py_function(
            _process_in_py,
            inp=[raw_speech],
            Tout=[self.compute_dtype, tf.bool],
        )
        num_frames = None
        if was_batched:
            input_features.set_shape([None, num_frames, self.feature_size])
            input_features_mask.set_shape([None, num_frames])
        else:
            input_features.set_shape([num_frames, self.feature_size])
            input_features_mask.set_shape([num_frames])
        input_features_mask = tf.cast(input_features_mask, dtype="int32")
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
                "mel_floor": self.mel_floor_arg,
                "per_bin_mean": self.per_bin_mean_arg,
                "per_bin_stddev": self.per_bin_stddev_arg,
                "padding_side": self.padding_side,
            }
        )
        return config
