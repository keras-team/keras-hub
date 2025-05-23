import keras

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone


@keras_hub_export("keras_hub.layers.MoonshineAudioConverter")
class MoonshineAudioConverter(AudioConverter):
    """Moonshine audio preprocessing layer.

    This layer processes raw audio waveforms for the Moonshine ASR model. Audio
    is formatted as a batched tensor at a 16kHz sample rate and validated for
    length (0.1 to 64 seconds). The layer handles padding and optional
    normalization. It does not contain trainable weights.

    Args:
        sampling_rate: int, optional. The audio sampling rate in Hz. Defaults to
            16,000.
        padding_value: float, optional. The value for padding. Defaults to 0.0.
        do_normalize: bool, optional. Whether to normalize inputs. Defaults to
            False.
        **kwargs: Additional keyword arguments passed to the base AudioConverter
            class for customizing the underlying preprocessing behavior.

    Call arguments:
        - `inputs`: The raw audio data to be processed. It should be a tensor of
          shape `(batch_size, time_steps, 1)` for mono audio. If the input has
          shape `(batch_size, time_steps)`, the layer will add the channel
          dimension.
        - `sampling_rate`: The sampling rate of the audio in Hz. If
          provided, it must match the expected sampling rate set during
          initialization (default is 16,000 Hz). If not provided, the expected
          sampling rate is taken from the initialization arguments.
        - `padding`: The padding strategy to apply. If provided, can be one of:
            - `"longest"`: If `pad_to_multiple_of` is set, pads the audio to
              make the time_steps dimension a multiple of `pad_to_multiple_of`.
            - `"max_length"`: Pads or truncates the audio to `max_length` time
              steps. If `pad_to_multiple_of` is set, the target length will be
              the smallest multiple of `pad_to_multiple_of` that is greater than
              or equal to `max_length`.
            - If not specified or `None`, no padding is applied.
        - `max_length`: The target number of time steps when `padding` is
          `"max_length"`. If not provided and `padding` is `"max_length"`, no
          padding or truncation is applied.
        - `pad_to_multiple_of`: If set, the padded time_steps will be a
          multiple of this value for the chosen padding strategy.

    Examples:
    ```python
    import keras
    from keras_hub.layers import MoonshineAudioConverter

    # Create a dummy audio input (1 second at 16kHz).
    dummy_audio = keras.ops.convert_to_tensor(
        [[0.1] * 16000],
        dtype="float32"
    )
    dummy_audio = keras.ops.expand_dims(dummy_audio, axis=-1)

    # Initialize the preprocessor.
    preprocessor = MoonshineAudioConverter(do_normalize=True)

    # Process the audio.
    processed_audio = preprocessor(dummy_audio)

    # Output shape.
    print(processed_audio.shape) # Expected: (1, 16000, 1) or padded length
    ```
    """

    # References:
    # Defined and formulated based on the UsefulSensors implementation of audio
    # preprocessing logic (https://github.com/usefulsensors/moonshine/blob/main/moonshine/transcribe.py).

    backbone_cls = MoonshineBackbone

    def __init__(
        self,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.do_normalize = do_normalize

    def call(
        self,
        inputs,
        sampling_rate=None,
        padding=None,
        max_length=None,
        pad_to_multiple_of=None,
    ):
        # Validate sampling rate.
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate {self.sampling_rate}, got "
                f"{sampling_rate}"
            )

        # Ensure inputs are (batch_size, time_steps, 1).
        input_shape = keras.ops.shape(inputs)
        input_rank = len(input_shape)
        if input_rank == 2:
            processed_inputs = keras.ops.expand_dims(inputs, axis=-1)
        elif input_rank == 3:
            processed_inputs = inputs
        else:
            raise ValueError(
                "Inputs must be mono audio: (batch_size, time_steps, 1)"
            )

        # Get original length and validate duration.
        current_shape = keras.ops.shape(processed_inputs)
        original_length = current_shape[1]
        duration = (
            keras.ops.cast(original_length, keras.backend.floatx())
            / self.sampling_rate
        )
        # Source: https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/transcribe.py#L20
        is_invalid_duration = keras.ops.logical_or(
            keras.ops.less(duration, 0.1), keras.ops.greater(duration, 64.0)
        )

        def print_warning_fn():
            import warnings

            warnings.warn(
                "Audio duration must be between 0.1 and 64 seconds. For "
                "transcribing longer segments, pre-segment your audio and "
                "provide shorter segments."
            )
            return keras.ops.convert_to_tensor(True, dtype="bool")

        is_tf_symbolic = (
            tf is not None
            and hasattr(processed_inputs, "graph")
            and hasattr(processed_inputs.graph, "as_graph_def")
        )
        use_tf_graph_ops = tf is not None and is_tf_symbolic
        if use_tf_graph_ops and keras.config.backend() != "torch":
            _ = tf.cond(
                is_invalid_duration,
                print_warning_fn,
                lambda: keras.ops.convert_to_tensor(False, dtype="bool"),
            )
        else:
            if keras.ops.convert_to_numpy(is_invalid_duration):
                print_warning_fn()

        # Handle padding.
        if padding == "longest":
            target_length = original_length
            if pad_to_multiple_of:
                target_length = (
                    (target_length + pad_to_multiple_of - 1)
                    // pad_to_multiple_of
                ) * pad_to_multiple_of

            needs_padding = keras.ops.greater(target_length, original_length)

            def pad_fn():
                padding_amount = target_length - original_length
                paddings = [[0, 0], [0, padding_amount], [0, 0]]
                if use_tf_graph_ops and keras.config.backend() != "tensorflow":
                    return tf.pad(
                        processed_inputs,
                        paddings,
                        mode="CONSTANT",
                        constant_values=float(self.padding_value),
                    )
                else:
                    return keras.ops.pad(
                        processed_inputs,
                        paddings,
                        mode="constant",
                        constant_values=self.padding_value,
                    )

            if use_tf_graph_ops and keras.config.backend() != "torch":
                processed_inputs = tf.cond(
                    needs_padding, pad_fn, lambda: processed_inputs
                )
            else:
                processed_inputs = keras.ops.cond(
                    needs_padding, pad_fn, lambda: processed_inputs
                )

        elif padding == "max_length" and max_length is not None:
            target_length_const = max_length
            if pad_to_multiple_of:
                target_length_const = (
                    (target_length_const + pad_to_multiple_of - 1)
                    // pad_to_multiple_of
                ) * pad_to_multiple_of

            needs_padding = keras.ops.less(original_length, target_length_const)
            needs_truncating = keras.ops.greater(
                original_length, target_length_const
            )

            def pad_fn():
                padding_amount = target_length_const - original_length
                paddings = [[0, 0], [0, padding_amount], [0, 0]]
                if use_tf_graph_ops and keras.config.backend() != "tensorflow":
                    return tf.pad(
                        processed_inputs,
                        paddings,
                        mode="CONSTANT",
                        constant_values=float(self.padding_value),
                    )
                else:
                    return keras.ops.pad(
                        processed_inputs,
                        paddings,
                        mode="constant",
                        constant_values=self.padding_value,
                    )

            def trunc_fn():
                if use_tf_graph_ops and keras.config.backend() != "tensorflow":
                    return processed_inputs[:, :target_length_const, :]
                else:
                    return keras.ops.slice(
                        processed_inputs,
                        [0, 0, 0],
                        [-1, target_length_const, -1],
                    )

            if use_tf_graph_ops and keras.config.backend() != "torch":
                processed_inputs = tf.cond(
                    needs_padding,
                    pad_fn,
                    lambda: tf.cond(
                        needs_truncating, trunc_fn, lambda: processed_inputs
                    ),
                )
            else:
                needs_padding = keras.ops.less(
                    original_length, target_length_const
                )
                needs_truncating = keras.ops.greater(
                    original_length, target_length_const
                )
                needs_padding_bool = keras.ops.convert_to_numpy(needs_padding)
                needs_truncating_bool = keras.ops.convert_to_numpy(
                    needs_truncating
                )

                if needs_padding_bool:
                    padding_amount = target_length_const - original_length
                    paddings = [[0, 0], [0, padding_amount], [0, 0]]
                    processed_inputs = keras.ops.pad(
                        processed_inputs,
                        paddings,
                        mode="constant",
                        constant_values=self.padding_value,
                    )
                elif needs_truncating_bool:
                    processed_inputs = processed_inputs[
                        :, :target_length_const, :
                    ]

        # Normalize if enabled.
        if self.do_normalize:
            mean = keras.ops.mean(processed_inputs, axis=1, keepdims=True)
            var = keras.ops.var(processed_inputs, axis=1, keepdims=True)
            processed_inputs = (processed_inputs - mean) / keras.ops.sqrt(
                var + 1e-7
            )

        return processed_inputs

    def compute_output_shape(self, input_shape):
        # [batch_size, time_steps] → [batch_size, time_steps, 1].
        if len(input_shape) == 2 or len(input_shape) == 3:
            return (input_shape[0], None, 1)
        else:
            raise ValueError("Input shape must be rank 2 or 3.")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "padding_value": self.padding_value,
                "do_normalize": self.do_normalize,
            }
        )
        return config
