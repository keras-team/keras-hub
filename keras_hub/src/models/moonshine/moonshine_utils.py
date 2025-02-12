import keras
from keras import layers
from keras import ops


# Specifically for PyTorch ops.reshape() compatibility.
# TODO: Check if we could do something with ops.reshape() in the PyTorch backend
# later so it does not require such a hack fix in the future.
def _to_concrete_int(dim):
    if isinstance(dim, keras.KerasTensor):
        if dim.device.type == "cuda":
            return int(dim.cpu().numpy())
        elif dim.device.type == "cpu":
            return int(dim.numpy())
        else:
            raise ValueError(f"Unsupported device type: {dim.device.type}")
    return int(dim)


def rotate_half(x):
    x_shape = [_to_concrete_int(dim) for dim in ops.shape(x)]
    new_shape = x_shape[:-1] + [-1, 2]

    x = ops.reshape(x, new_shape)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x_rotated = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(x_rotated, x_shape)


def apply_rotary_pos_emb(t, freqs):
    # t: tensor to rotate; freqs: precomputed frequencies.
    rot_dim = _to_concrete_int(ops.shape(freqs)[-1])
    seq_len = _to_concrete_int(ops.shape(t)[-3])

    orig_dtype = ops.dtype(t)
    freqs = freqs[-seq_len:, :]

    # Convert shape to concrete integers for reshape.
    freq_shape = [
        _to_concrete_int(dim) for dim in [freqs.shape[0], 1, freqs.shape[1]]
    ]
    freqs = ops.reshape(freqs, freq_shape)

    # Split tensor into rotary and non-rotary parts.
    t_rot, t_nonrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rotated = t_rot * ops.cos(freqs) + rotate_half(t_rot) * ops.sin(freqs)
    out = ops.concatenate((t_rotated, t_nonrot), axis=-1)
    return ops.cast(out, orig_dtype)


class InvFreqInitializer(keras.initializers.Initializer):
    def __init__(self, dim, base):
        self.dim = dim
        self.base = base

    def __call__(self, shape, dtype=None):
        # Compute inverse frequencies for rotary embeddings.
        positions = ops.arange(0, self.dim, 2, dtype="float32")
        return 1.0 / (self.base ** (positions / self.dim))


class RotaryEmbedding(layers.Layer):
    def __init__(self, dim, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # Create non-trainable weights for inverse frequency.
        self.inv_freq = self.add_weight(
            shape=(dim // 2,),
            initializer=InvFreqInitializer(dim, base),
            trainable=False,
            name="inv_freq",
        )

    def call(self, position_ids):
        # Position_ids shape: (seq_len,).
        freqs = ops.einsum("i,j->ij", position_ids, self.inv_freq)
        emb = ops.repeat(freqs, 2, axis=-1)
        return emb


class Arange(layers.Layer):
    def call(self, inputs):
        # Inputs is expected to be a tensor where the first element is a scalar
        # length.
        length = _to_concrete_int(inputs[0])
        return ops.arange(length)

    def compute_output_shape(self, input_shape):
        return (None,)
