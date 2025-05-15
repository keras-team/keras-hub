from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty

try:
    import tensorflow as tf
except ImportError:
    tf = None

import keras


@keras_hub_export("keras_hub.layers.AudioConverter")
class AudioConverter(PreprocessingLayer):
    """Convert raw audio for models that support audio input.

    This class converts from raw audio tensors of any length, to preprocessed
    audio for pretrained model inputs. It is meant to be a convenient way to
    write custom preprocessing code that is not model specific. This layer
    should be instantiated via the `from_preset()` constructor, which will
    create the correct subclass of this layer for the model preset.

    The layer will take as input a raw audio tensor with shape `(batch_size,
    num_samples)`, and output a preprocessed audio input for modeling. The exact
    structure of the preprocessed input will vary per model. Preprocessing
    will often include computing a spectogram of the raw audio signal.

    Examples:
    ```python
    # Load an audio converter from a preset.
    converter = keras_hub.layers.AudioConverter.from_preset("whisper_base_en")
    # Convert some raw audio input.
    converter(np.ones(2, 1_000))
    ```
    """

    backbone_cls = None

    @classproperty
    def presets(cls):
        """List built-in presets for an `AudioConverter` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.layers.AudioConverter` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'whisper_base_en'`
        2. a Kaggle Models handle like
           `'kaggle://user/whisper/keras/whisper_base_en'`
        3. a Hugging Face handle like `'hf://user/whisper_base_en'`
        4. a path to a local preset directory like `'./whisper_base_en'`

        You can run `cls.presets.keys()` to list all built-in presets available
        on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.AudioConverter.from_preset()`, or from a
        model class like `keras_hub.models.WhisperAudioConverter.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load an audio converter from a preset.
        converter = keras_hub.layers.AudioConverter.from_preset(
            "whisper_base_en"
        )
        # Convert some raw mono channel audio input.
        converter(np.ones(2, 1_000))
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_audio_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save audio converter to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_audio_converter(self)

    def _is_tf_symbolic(self, tensor):
        return (
            tf is not None
            and hasattr(tensor, "graph")
            and hasattr(tensor.graph, "as_graph_def")
        )

    def _squeeze(self, tensor, axis=None):
        """Avoid issues with keras.ops.squeeze() with tf.squeeze() if
        needed."""
        if (
            tf is not None
            and isinstance(tensor, tf.Tensor)
            and keras.config.backend() != "tensorflow"
        ):
            return tf.squeeze(tensor, axis=axis)
        else:
            return keras.ops.squeeze(tensor, axis=axis)

    def _use_tf_graph_ops(self, tensor):
        return tf is not None and self._is_tf_symbolic(tensor)

    def _pad(self, tensor, paddings, mode="constant", constant_values=0.0):
        """Pad a tensor, using tf.pad only for symbolic tensors, the rest use
        the backend-agnostic keras.ops.pad()."""
        if (
            self._use_tf_graph_ops(tensor)
            and keras.config.backend() != "tensorflow"
        ):
            tf_mode = mode.upper()
            if tf_mode == "CONSTANT":
                tf_constant_values = float(constant_values)
                return tf.pad(
                    tensor,
                    paddings,
                    mode=tf_mode,
                    constant_values=tf_constant_values,
                )
            else:
                return tf.pad(tensor, paddings, mode=tf_mode)
        else:
            if mode == "constant":
                return keras.ops.pad(
                    tensor, paddings, mode=mode, constant_values=constant_values
                )
            else:
                return keras.ops.pad(tensor, paddings, mode=mode)

    def _cond(self, condition, true_fn, false_fn, tensor):
        """Conditional execution based on backend and tensor type."""
        if self._use_tf_graph_ops(tensor) and keras.config.backend() != "torch":
            return tf.cond(condition, true_fn, false_fn)
        else:
            if keras.ops.convert_to_numpy(condition):
                return true_fn()
            else:
                return false_fn()

    def _stft(self, audio, frame_length, frame_step, fft_length):
        """Compute the Short-Time Fourier Transform (STFT)."""
        if tf is None:
            raise ImportError("TensorFlow is required for STFT computation.")
        return tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
        )

    def _normalize(self, audio, axis=1):
        """Normalize audio by subtracting mean and dividing by standard
        deviation."""
        mean = keras.ops.mean(audio, axis=axis, keepdims=True)
        var = keras.ops.var(audio, axis=axis, keepdims=True)
        return (audio - mean) / keras.ops.sqrt(var + 1e-7)

    def _compute_magnitudes(self, stft):
        """Compute magnitudes from STFT output."""
        return keras.ops.square(keras.ops.abs(stft))

    def _apply_mel_filters(self, magnitudes, mel_filters):
        """Apply mel filters to magnitudes."""
        return keras.ops.matmul(magnitudes, mel_filters)

    def _log_transform(self, spectrogram, min_value=1e-10):
        """Apply log transform with a minimum value for numerical stability."""
        spectrogram = keras.ops.maximum(spectrogram, min_value)
        return keras.ops.log(spectrogram) / keras.ops.log(
            keras.ops.convert_to_tensor(10.0)
        )

    def _dynamic_range_compression(self, spectrogram, threshold=8.0):
        if tf is None:
            raise ImportError(
                "TensorFlow is required for dynamic range compression."
            )
        log_spec_shape = tf.shape(spectrogram)
        max_value_minus_threshold = tf.math.subtract(
            tf.math.reduce_max(spectrogram, axis=[1, 2]),
            tf.cast(threshold, dtype=spectrogram.dtype),
        )
        max_value_minus_threshold = tf.expand_dims(
            max_value_minus_threshold, axis=1
        )
        max_value_minus_threshold = tf.repeat(
            max_value_minus_threshold,
            repeats=log_spec_shape[1] * log_spec_shape[2],
            axis=1,
        )
        max_value_minus_threshold = tf.reshape(
            max_value_minus_threshold, shape=log_spec_shape
        )
        return tf.maximum(spectrogram, max_value_minus_threshold)

    def _compute_log_mel_spectrogram(
        self,
        audio,
        frame_length,
        frame_step,
        fft_length,
        mel_filters,
        pre_padding=None,
        post_padding=None,
        normalize=False,
        compress=False,
        compression_threshold=8.0,
    ):
        """Compute the log-mel spectrogram."""
        if tf is None:
            raise ImportError(
                "TensorFlow is required for computing the log mel spectrogram."
            )
        audio = tf.cast(audio, self.compute_dtype)
        # Use "reflection" padding - `tf.signal.stft` uses symmetric padding
        # internally.
        if pre_padding is not None or post_padding is not None:
            paddings = [
                [0, 0],
                [
                    pre_padding if pre_padding is not None else 0,
                    post_padding if post_padding is not None else 0,
                ],
            ]
            audio = self._pad(
                audio,
                paddings=paddings,
                mode="reflect",
            )
        # Compute the mel spectrogram.
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
        )
        magnitudes = tf.square(tf.abs(stft[:, :-1, :]))
        mel_spec = tf.matmul(
            magnitudes,
            mel_filters,
        )
        # Clamp the values to a minimum value of 1e-10. This is done to avoid
        # taking the log of 0, i.e., for numerical stability.
        mel_spec = tf.maximum(mel_spec, 1e-10)
        # Calculate the log mel spectrogram.
        log_spec = tf.math.log(mel_spec) / tf.math.log(
            tf.constant(10, dtype=mel_spec.dtype)
        )
        if compress:
            # Dynamic range compression.
            log_spec = self._dynamic_range_compression(
                log_spec, threshold=compression_threshold
            )
        if normalize:
            # Normalization.
            type_cast_four = tf.cast(4, dtype=log_spec.dtype)
            log_spec = tf.math.divide(
                tf.math.add(log_spec, type_cast_four), type_cast_four
            )

        return log_spec

    def _process_audio_tensor(self, audio, num_samples):
        """Process the input audio tensor."""
        if tf is None:
            raise ImportError(
                "TensorFlow is required for processing the input audio tensor."
            )
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            audio = tf.convert_to_tensor(audio)
        rank_1_input = audio.shape.rank == 1
        if rank_1_input:
            audio = tf.expand_dims(audio, 0)
        if isinstance(audio, tf.Tensor):
            audio = tf.RaggedTensor.from_tensor(audio)
        # Pad audio.
        audio_shape = audio.shape.as_list()
        audio_shape[-1] = num_samples
        audio = audio.to_tensor(shape=audio_shape)
        return audio, rank_1_input
