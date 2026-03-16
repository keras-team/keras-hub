import math

import keras
import numpy as np
from keras import ops

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
            "longest"), or `False` (no padding). Defaults to `"longest"`.
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

    # Create a simple audio signal (1 second of 440 Hz sine wave).
    audio = np.sin(
        2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32)
    )

    # Initialize the audio converter
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

    # Convert audio to log-mel spectrogram.
    features, mask = converter(audio)
    print(features.shape)  # (num_frames, 128)
    print(mask.shape)      # (num_frames,)

    # Convert a batch of audio with padding.
    audio_1 = np.sin(
        2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32)
    )
    audio_2 = np.sin(
        2 * np.pi * 880 * np.linspace(0, 0.5, 8000, dtype=np.float32)
    )
    features, mask = converter(
        [audio_1, audio_2],
        padding="longest",
        pad_to_multiple_of=128,
    )
    print(features.shape)  # (2, num_frames, 128)
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
        self.mel_floor = ops.convert_to_tensor(
            mel_floor, dtype=self.compute_dtype
        )
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length
        hann_arange = ops.arange(self.frame_length, dtype=self.compute_dtype)
        self.window = 0.5 * (
            1 - ops.cos(2 * np.pi * hann_arange / self.frame_length)
        )
        self.mel_filters = self._create_filterbank_matrix(
            n_freqs=self.fft_length // 2 + 1,
            f_min=min_frequency,
            f_max=max_frequency,
            n_mels=feature_size,
            sample_rate=self.sampling_rate,
            fft_length=fft_length,
        )
        if per_bin_mean is not None:
            self.per_bin_mean = ops.reshape(
                ops.convert_to_tensor(per_bin_mean, dtype=self.compute_dtype),
                (1, 1, feature_size),
            )
        else:
            self.per_bin_mean = None
        if per_bin_stddev is not None:
            self.per_bin_stddev = ops.reshape(
                ops.convert_to_tensor(per_bin_stddev, dtype=self.compute_dtype),
                (1, 1, feature_size),
            )
        else:
            self.per_bin_stddev = None
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.built = True

    def audio_shape(self):
        """Returns the preprocessed size of a single audio sample."""
        return (None, self.feature_size)

    def _create_filterbank_matrix(
        self,
        n_freqs,
        f_min,
        f_max,
        n_mels,
        sample_rate,
        fft_length,
    ):
        all_freqs = ops.cast(ops.arange(n_freqs), dtype=self.compute_dtype) * (
            sample_rate / fft_length
        )
        m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        f_pts = ops.convert_to_tensor(f_pts, dtype=self.compute_dtype)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = ops.expand_dims(f_pts, 0) - ops.expand_dims(all_freqs, 1)
        zero = ops.zeros(1, dtype=self.compute_dtype)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = ops.maximum(zero, ops.minimum(down_slopes, up_slopes))
        return ops.convert_to_tensor(fb, dtype=self.compute_dtype)

    def _extract_spectrogram(self, waveform, attention_mask):
        waveform = ops.cast(waveform, dtype=self.compute_dtype)
        if self.dither > 0.0:
            waveform = waveform + self.dither * keras.random.normal(
                ops.shape(waveform), dtype=waveform.dtype
            )
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor
        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_sample = waveform[:, :1] * (1.0 - self.preemphasis)
                rest_of_samples = (
                    waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
                )
                waveform = ops.concatenate(
                    [first_sample, rest_of_samples], axis=-1
                )
            else:
                waveform = ops.concatenate(
                    [
                        waveform[:, :1],
                        waveform[:, 1:] - self.preemphasis * waveform[:, :-1],
                    ],
                    axis=-1,
                )

        # Manual framing to replace tf.signal.frame
        waveform_length = ops.shape(waveform)[1]
        num_frames = (
            waveform_length - self.frame_length
        ) // self.hop_length + 1

        # Create frame indices
        frame_starts = ops.arange(num_frames) * self.hop_length
        frame_indices = ops.expand_dims(frame_starts, axis=-1) + ops.arange(
            self.frame_length
        )

        # Gather frames
        frames = ops.take(waveform, frame_indices, axis=1)

        # Apply window
        frames = frames * self.window

        # Pad frames for FFT
        pad_length = self.fft_length - self.frame_length
        paddings = [[0, 0], [0, 0], [0, pad_length]]
        frames = ops.pad(frames, paddings)

        # Compute RFFT
        real_part, imag_part = ops.rfft(frames, fft_length=self.fft_length)
        magnitude_spec = ops.sqrt(real_part**2 + imag_part**2)

        mel_spec = ops.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = ops.log(ops.maximum(mel_spec, self.mel_floor))
        if self.per_bin_mean is not None:
            log_mel_spec = log_mel_spec - self.per_bin_mean
        if self.per_bin_stddev is not None:
            log_mel_spec = log_mel_spec / self.per_bin_stddev
        mel_spectrogram = ops.squeeze(log_mel_spec, axis=0)
        mask = ops.cast(attention_mask[:: self.hop_length], dtype="bool")
        return mel_spectrogram, mask[: ops.shape(mel_spectrogram)[0]]

    def call(
        self,
        raw_speech,
        padding="longest",
        max_length=480000,
        truncation=True,
        pad_to_multiple_of=128,
        return_attention_mask=True,
    ):
        # Convert input to tensor and determine if batched
        if not ops.is_tensor(raw_speech):
            if isinstance(raw_speech, (list, tuple)):
                # Check if it's a list of sequences (batched) or single sequence
                if len(raw_speech) > 0 and isinstance(
                    raw_speech[0], (list, np.ndarray)
                ):
                    # Batched: list of sequences
                    was_batched = True
                    # Convert each to tensor and find max length for padding
                    tensors = [
                        ops.reshape(
                            ops.convert_to_tensor(s, dtype=self.compute_dtype),
                            [-1],
                        )
                        for s in raw_speech
                    ]
                    lengths = [ops.shape(t)[0] for t in tensors]

                    if padding == "longest" or padding is True:
                        max_length = max(lengths)
                        if pad_to_multiple_of is not None:
                            remainder = max_length % pad_to_multiple_of
                            if remainder != 0:
                                max_length = max_length + (
                                    pad_to_multiple_of - remainder
                                )

                    # Truncate and pad each tensor
                    padded_tensors = []
                    masks = []
                    for t, orig_len in zip(tensors, lengths):
                        # Truncate
                        if truncation and max_length is not None:
                            t = t[:max_length]
                            orig_len = min(orig_len, max_length)

                        # Pad
                        if max_length is not None:
                            current_len = ops.shape(t)[0]
                            pad_amount = max_length - current_len
                            if self.padding_side == "right":
                                t = ops.pad(
                                    t,
                                    [[0, pad_amount]],
                                    constant_values=self.padding_value,
                                )
                            else:
                                t = ops.pad(
                                    t,
                                    [[pad_amount, 0]],
                                    constant_values=self.padding_value,
                                )

                        # Create mask
                        final_len = ops.shape(t)[0]
                        mask = ops.cast(
                            ops.arange(final_len) < orig_len, dtype="int32"
                        )

                        padded_tensors.append(t)
                        masks.append(mask)

                    raw_speech = ops.stack(padded_tensors, axis=0)
                    attention_masks = ops.stack(masks, axis=0)
                else:
                    # Single sequence
                    was_batched = False
                    raw_speech = ops.reshape(
                        ops.convert_to_tensor(
                            raw_speech, dtype=self.compute_dtype
                        ),
                        [1, -1],
                    )
            else:
                # Single array/value
                was_batched = False
                raw_speech = ops.reshape(
                    ops.convert_to_tensor(raw_speech, dtype=self.compute_dtype),
                    [1, -1],
                )
        else:
            # Already a tensor
            ndim = len(raw_speech.shape)
            was_batched = ndim > 1
            if not was_batched:
                raw_speech = ops.reshape(raw_speech, [1, -1])

        # Handle tensor input (create masks if not already created)
        if "attention_masks" not in locals():
            batch_size = ops.shape(raw_speech)[0]
            seq_length = ops.shape(raw_speech)[1]

            # Determine target length
            if padding == "longest" or padding is True:
                target_length = seq_length
                if pad_to_multiple_of is not None:
                    remainder = target_length % pad_to_multiple_of
                    if remainder != 0:
                        target_length = target_length + (
                            pad_to_multiple_of - remainder
                        )
                max_length = target_length

            # Truncate if needed
            if (
                truncation
                and max_length is not None
                and seq_length > max_length
            ):
                raw_speech = raw_speech[:, :max_length]
                seq_length = max_length

            # Pad if needed
            if max_length is not None and seq_length < max_length:
                pad_amount = max_length - seq_length
                if self.padding_side == "right":
                    raw_speech = ops.pad(
                        raw_speech,
                        [[0, 0], [0, pad_amount]],
                        constant_values=self.padding_value,
                    )
                else:
                    raw_speech = ops.pad(
                        raw_speech,
                        [[0, 0], [pad_amount, 0]],
                        constant_values=self.padding_value,
                    )

            # Create attention masks
            final_length = ops.shape(raw_speech)[1]
            attention_masks = ops.ones(
                (batch_size, final_length), dtype="int32"
            )

        # Process spectrogram extraction using map
        def process_single_sample(args):
            waveform, mask = args
            waveform = ops.expand_dims(
                waveform, axis=0
            )  # Add batch dim for _extract_spectrogram
            features, feature_mask = self._extract_spectrogram(waveform, mask)
            return features, feature_mask

        results = ops.map(process_single_sample, (raw_speech, attention_masks))

        input_features, input_features_mask = results

        # Remove batch dimension if input wasn't batched
        if not was_batched:
            input_features = input_features[0]
            input_features_mask = input_features_mask[0]

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
