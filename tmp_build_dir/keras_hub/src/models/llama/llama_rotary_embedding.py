import math

from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding


class LlamaRotaryEmbedding(RotaryEmbedding):
    """Rotary positional encoding layer.

    This layer encodes absolute positional information with a rotation
    matrix. It calculates the rotary encoding with a mix of sine and
    cosine functions with geometrically increasing wavelengths.
    Defined and formulated in
    [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
    The input must be a tensor with shape a sequence dimension and a feature
    dimension. Typically, this will either an input with shape
    `(batch_size, sequence_length, feature_length)` or
    `(batch_size, sequence_length, num_heads, feature_length)`.
    This layer will return a new tensor with the rotary embedding applied to
    the input tensor.
    It is extended from `RotaryEmbedding` layer in `keras_hub.layers`.
    It has additional smoothening and interpolation for some frequency ranges.

    Args:
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves. Defaults to `10000`.
        position_scaling_factor: float. The scaling factor used to scale
            positions of the tokens. Defaults to `1.0`.
        frequency_adjustment_factor: float. The scaling factor used to scale the
            inverse frequencies. Defaults to `None`.
        low_freq_factor: float. The low frequency scaling factor.
            Defaults to `None`.
        high_freq_factor: float. The high frequency scaling factor.
            Defaults to `None`.
        pretraining_sequence_length: int. Used for Llama3.1+, the original
            context length at time of pretraining. Defaults to `None`.
        sequence_axis: int. Sequence axis in the input tensor.
        feature_axis: int. Feature axis in the input tensor.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.

    Examples:

    ```python
    batch_size = 16
    feature_length = 18
    sequence_length = 256
    num_heads = 8

    # No multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, feature_length))
    rot_emb_layer = RotaryEmbedding()
    tensor_rot = rot_emb_layer(tensor)

    # With multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, num_heads, feature_length))
    tensor_rot = rot_emb_layer(tensor)
    ```

    References:
     - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
    """

    def __init__(
        self,
        max_wavelength=10000,
        position_scaling_factor=1.0,
        sequence_axis=1,
        feature_axis=-1,
        frequency_adjustment_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        pretraining_sequence_length=None,
        **kwargs,
    ):
        super().__init__(
            max_wavelength=max_wavelength,
            scaling_factor=position_scaling_factor,
            sequence_axis=sequence_axis,
            feature_axis=feature_axis,
            **kwargs,
        )
        self.max_wavelength = max_wavelength
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis
        self.position_scaling_factor = position_scaling_factor
        self.frequency_adjustment_factor = frequency_adjustment_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.pretraining_sequence_length = pretraining_sequence_length

        grouped_args = [
            low_freq_factor,
            high_freq_factor,
            frequency_adjustment_factor,
            pretraining_sequence_length,
        ]
        args_none = [x is None for x in grouped_args]
        if any(args_none) and not all(args_none):
            raise ValueError(
                "Either all of `low_freq_factor`,`high_freq_factor`, "
                "`frequency_adjustment_factor` and "
                "`pretraining_sequence_length` should be set, or all of should"
                " be set `None`."
            )
        self.built = True

    def _get_inverse_freq(self, rotary_dim):
        freq_range = ops.divide(
            ops.arange(0, rotary_dim, 2, dtype="float32"),
            ops.cast(rotary_dim, "float32"),
        )
        inverse_freq = 1.0 / (self.max_wavelength**freq_range)

        # From llama3.1+ we have additional smoothening and interpolation.
        # low_freq_factor, high_freq_factor, pretraining_sequence_length,
        # frequency_adjustment_factor are all set at once so it is fine.
        if self.low_freq_factor is not None:
            low_freq_wavelen = (
                self.pretraining_sequence_length / self.low_freq_factor
            )
            high_freq_wavelen = (
                self.pretraining_sequence_length / self.high_freq_factor
            )
            wavelen = 2 * math.pi / inverse_freq

            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            inverse_freq = ops.where(
                ops.greater(wavelen, low_freq_wavelen),
                (inverse_freq / self.frequency_adjustment_factor),
                inverse_freq,
            )

            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (
                (self.pretraining_sequence_length / wavelen)
                - self.low_freq_factor
            ) / (self.high_freq_factor - self.low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * (
                inverse_freq / self.frequency_adjustment_factor
            ) + (smooth_factor * inverse_freq)
            is_medium_freq = ops.logical_and(
                ops.greater_equal(wavelen, high_freq_wavelen),
                ops.less_equal(wavelen, low_freq_wavelen),
            )

            inverse_freq = ops.where(
                is_medium_freq, smoothed_inv_freq, inverse_freq
            )

        return inverse_freq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "sequence_axis": self.sequence_axis,
                "feature_axis": self.feature_axis,
                "position_scaling_factor": self.position_scaling_factor,
                "frequency_adjustment_factor": self.frequency_adjustment_factor,
                "low_freq_factor": self.low_freq_factor,
                "high_freq_factor": self.high_freq_factor,
                "original_max_embeddings": self.pretraining_sequence_length,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
