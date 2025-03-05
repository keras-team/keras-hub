import keras_hub
from keras import ops
import math

from keras_hub.src.api_export import keras_hub_export


class Llama31RotaryEmbedding(keras_hub.layers.RotaryEmbedding):
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

    Args:
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves.
        scaling_factor: float. The scaling factor used to scale positions of
            the tokens.
        sequence_axis: int. Sequence axis in the input tensor.
        feature_axis: int. Feature axis in the input tensor.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
s
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
        scaling_factor=1.0,
        sequence_axis=1,
        feature_axis=-1,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        old_context_len=8192,




        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis
        self.scaling_factor = scaling_factor
        self.factor=factor=8,
        self.llow_freq_factor=low_freq_factor,
        self.high_freq_factor=high_freq_factor,
        self.old_context_len=old_context_len,
        self.built = True

    def _get_inverse_freq(self, rotary_dim):
        freq_range = ops.divide(
            ops.arange(0, rotary_dim, 2, dtype="float32"),
            ops.cast(rotary_dim, "float32"),
        )
        inverse_freq = 1.0 / (self.max_wavelength**freq_range)

        factor = 8
        old_context_len = self.old_context_len
        low_freq_factor = self.llow_freq_factor
        high_freq_factor = self.high_freq_factor
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inverse_freq
        

        inv_freq_llama = keras.ops.where(
            ops.greater(wavelen , low_freq_wavelen), inverse_freq / factor, inverse_freq
        )

        # # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq=ops.bitwise_and(ops.cast(ops.greater_equal(wavelen,high_freq_wavelen),dtype='int8'),
                                       ops.cast(ops.less_equal(wavelen,low_freq_wavelen),dtype='int8'))

        # inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq_llama = ops.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        ops.cast(inv_freq_llama,'float32')

        return inv_freq_llama
