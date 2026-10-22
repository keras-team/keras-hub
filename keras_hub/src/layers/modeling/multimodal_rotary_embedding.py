import functools
import math

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding


@functools.lru_cache(maxsize=None)
def _channel_selector(mrope_section):
    # One-hot `(3, half_dim)` picking the T, H or W channel for each
    # frequency, as plain Python floats so it is safe to share across traces.
    half_dim = sum(mrope_section)
    channel = [0] * half_dim
    for dim in (1, 2):
        for index in range(dim, min(mrope_section[dim] * 3, half_dim), 3):
            channel[index] = dim
    return tuple(tuple(float(c == dim) for c in channel) for dim in range(3))


@keras_hub_export("keras_hub.layers.MultimodalRotaryEmbedding")
class MultimodalRotaryEmbedding(RotaryEmbedding):
    """Interleaved multimodal rotary position embedding (M-RoPE).

    M-RoPE gives every token three positions, temporal, height and width,
    instead of one. Each rotary frequency takes its angle from one of the
    three. Following Qwen3-VL, the channels are interleaved over the
    frequencies as `T, H, W, T, H, W, ...`: frequency `i` uses H when
    `i % 3 == 1` and `i < 3 * mrope_section[1]`, W when `i % 3 == 2` and
    `i < 3 * mrope_section[2]`, and T otherwise. Text tokens have the same
    value on all three channels, which reduces to standard RoPE.

    The cos and sin tables only depend on the positions, so a model can
    build them once with `compute_embeddings()` and pass them to every
    attention layer as `rotary_embeddings`.

    The rotation covers the first `2 * sum(mrope_section)` features. Any
    features after that are passed through unchanged, which supports
    partial rotary embeddings.

    Args:
        mrope_section: tuple of 3 ints. The number of rotary frequencies
            taken from the temporal, height and width positions.
        max_wavelength: int. The maximum angular wavelength of the sine and
            cosine curves. Defaults to `10000`.
        scaling_factor: float. The scaling factor used to scale positions.
            Defaults to `1.0`.
        attention_scaling: float. A factor applied to the cos and sin
            tables. With `rope_type="yarn"`, the YaRN temperature is applied
            on top of it. Defaults to `1.0`.
        **kwargs: other keyword arguments passed to `RotaryEmbedding`,
            including `rope_type`, `name` and `dtype`.

    Call arguments:
        inputs: Tensor of shape `(batch, sequence_length, ..., features)`,
            typically `(batch, sequence_length, num_heads, head_dim)`.
        position_ids: Optional int Tensor of shape
            `(batch, 3, sequence_length)` with the temporal, height and width
            positions of `inputs`. When `None`, all three channels use
            `start_index + arange(sequence_length)`.
        start_index: int or int Tensor. The position of the first token when
            `position_ids` is `None`. Defaults to `0`.
        rotary_embeddings: Optional `(cos, sin)` tuple from
            `compute_embeddings()`. When given, `position_ids` and
            `start_index` are ignored.

    Example:

    ```python
    rotary = keras_hub.layers.MultimodalRotaryEmbedding(
        mrope_section=(24, 20, 20), max_wavelength=5000000
    )
    query = np.random.normal(size=(2, 10, 16, 128))
    key = np.random.normal(size=(2, 10, 8, 128))
    position_ids = np.tile(np.arange(10), (2, 3, 1))

    # Rotate one tensor.
    query = rotary(query, position_ids=position_ids)

    # Build the tables once and reuse them.
    rotary_embeddings = rotary.compute_embeddings(position_ids)
    key = rotary(key, rotary_embeddings=rotary_embeddings)
    ```
    """

    def __init__(
        self,
        mrope_section,
        max_wavelength=10000,
        scaling_factor=1.0,
        attention_scaling=1.0,
        **kwargs,
    ):
        super().__init__(
            max_wavelength=max_wavelength,
            scaling_factor=scaling_factor,
            **kwargs,
        )
        if len(mrope_section) != 3:
            raise ValueError(
                "`mrope_section` must have 3 entries (temporal, height, "
                f"width). Received: mrope_section={mrope_section}"
            )
        if self.sequence_axis != 1 or self.feature_axis != -1:
            raise ValueError(
                "`MultimodalRotaryEmbedding` expects inputs of shape "
                "`(batch, sequence_length, ..., features)`. Received: "
                f"sequence_axis={self.sequence_axis}, "
                f"feature_axis={self.feature_axis}"
            )
        self.mrope_section = tuple(mrope_section)
        self.attention_scaling = attention_scaling
        self.rotary_dim = 2 * sum(self.mrope_section)

    def compute_embeddings(
        self, position_ids=None, sequence_length=None, start_index=0
    ):
        """Return the float32 `(cos, sin)` tables.

        Args:
            position_ids: Optional int Tensor `(batch, 3, sequence_length)`.
            sequence_length: int. Required when `position_ids` is `None`.
            start_index: int or int Tensor. The position of the first token
                when `position_ids` is `None`.

        Returns:
            A `(cos, sin)` tuple, each of shape
            `(batch or 1, sequence_length, 1, 2 * sum(mrope_section))`.
        """
        inverse_freq = self._get_inverse_freq(self.rotary_dim)
        if position_ids is None:
            if sequence_length is None:
                raise ValueError(
                    "Pass `position_ids` or `sequence_length`. Received "
                    "neither."
                )
            positions = ops.arange(sequence_length, dtype="float32")
            positions = positions + ops.cast(start_index, "float32")
            # `(1, 1, seq)`: the three channels share the text position.
            positions = positions[None, None, :]
        else:
            positions = ops.cast(position_ids, "float32")
        # `(batch, 3, seq, half_dim)`, then pick one channel per frequency.
        freqs = positions[..., None] * inverse_freq
        selector = ops.convert_to_tensor(
            _channel_selector(self.mrope_section), dtype="float32"
        )
        freqs = ops.sum(freqs * selector[None, :, None, :], axis=1)
        embedding = ops.concatenate([freqs, freqs], axis=-1)
        cos = ops.expand_dims(ops.cos(embedding), axis=2)
        sin = ops.expand_dims(ops.sin(embedding), axis=2)
        scale = self.attention_scaling
        if self.rope_type == "yarn":
            # The YaRN temperature, as in `RotaryEmbedding`.
            if self.attention_factor is not None:
                scale *= self.attention_factor
            else:
                scale *= 0.1 * math.log(self.scaling_factor) + 1.0
        if scale != 1.0:
            cos = cos * scale
            sin = sin * scale
        return cos, sin

    def call(
        self,
        inputs,
        position_ids=None,
        start_index=0,
        rotary_embeddings=None,
    ):
        if rotary_embeddings is None:
            rotary_embeddings = self.compute_embeddings(
                position_ids,
                sequence_length=ops.shape(inputs)[1],
                start_index=start_index,
            )
        cos, sin = rotary_embeddings
        # The tables carry one head axis. Drop it for `(batch, seq, features)`.
        if len(inputs.shape) == 3:
            cos, sin = cos[:, :, 0, :], sin[:, :, 0, :]
        cos = ops.cast(cos, inputs.dtype)
        sin = ops.cast(sin, inputs.dtype)
        if inputs.shape[-1] == self.rotary_dim:
            return self._rotate(inputs, cos, sin)
        rotated = self._rotate(inputs[..., : self.rotary_dim], cos, sin)
        return ops.concatenate(
            [rotated, inputs[..., self.rotary_dim :]], axis=-1
        )

    def _rotate(self, x, cos, sin):
        x1, x2 = ops.split(x, 2, axis=-1)
        half_rotated = ops.concatenate([-x2, x1], axis=-1)
        return x * cos + half_rotated * sin

    def compute_output_spec(self, inputs, *args, **kwargs):
        return keras.KerasTensor(inputs.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mrope_section": self.mrope_section,
                "attention_scaling": self.attention_scaling,
            }
        )
        return config
