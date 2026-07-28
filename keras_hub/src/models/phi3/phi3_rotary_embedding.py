import math

from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding


class Phi3SuScaledRotaryEmbedding(RotaryEmbedding):
    """SuRotary positional encoding layer.

    Args:
        inverese_freq_short_factor List[float]: List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is smaller than `original_max_sequence_length`.
        inverese_freq_long_factor List[float]: List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is larger than `original_max_sequence_length`.
        max_sequence_length: int. The maximum sequence length that this
            model might ever be used with.
        pretraining_sequence_length: int. The maximum sequence length that
            this model was pretrained with.
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves.

    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.
        positions: Optional tensor of absolute positions, shaped
            `(sequence_length,)` or `(batch_size, sequence_length)`, taking
            precedence over `start_index`.

    References:
     - [Phi-3-mini-128k-instruct original implementation](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/0693e0b867d29e7318280ddd3ff9d5e66698f488/modeling_phi3.py#L142)
    """

    def __init__(
        self,
        inverese_freq_short_factor,
        inverese_freq_long_factor,
        max_sequence_length=4096,
        pretraining_sequence_length=4096,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(max_wavelength=max_wavelength, **kwargs)
        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length

        scaling_factor = (
            self.max_sequence_length / self.pretraining_sequence_length
        )
        if scaling_factor <= 1.0:
            self.embedding_scaling_factor = 1.0
        else:
            self.embedding_scaling_factor = math.sqrt(
                1
                + math.log(scaling_factor)
                / math.log(self.pretraining_sequence_length)
            )

        self.inverese_freq_short_factor = inverese_freq_short_factor
        self.inverese_freq_long_factor = inverese_freq_long_factor

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        batch_axis = 0
        sequence_axis = 1
        feature_axis = len(inputs.shape) - 1

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
            positions = ops.expand_dims(positions, axis=batch_axis)
        else:
            positions = ops.cast(positions, "float32")
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=batch_axis)

        # Multiply inverse_freq by a factor, chosen by how far into the
        # sequence these positions sit rather than by how many tokens this
        # call carries. Cached decoding passes one token at a time, so
        # keying on the input's length would apply short-context
        # frequencies at every generated position, however long the
        # sequence has grown.
        sequence_length = ops.max(positions) + 1.0
        inverse_freq = ops.where(
            sequence_length > self.pretraining_sequence_length,
            ops.divide(
                inverse_freq,
                ops.convert_to_tensor(self.inverese_freq_long_factor),
            ),
            ops.divide(
                inverse_freq,
                ops.convert_to_tensor(self.inverese_freq_short_factor),
            ),
        )

        freq = ops.einsum("bi,j->bij", positions, inverse_freq)
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(
            embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2)
        )

        # Reshape the embedding to be broadcastable with input shape.
        for axis in range(len(inputs.shape)):
            if axis not in (batch_axis, sequence_axis, feature_axis):
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(
            ops.cos(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        sin_emb = ops.cast(
            ops.sin(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        return cos_emb, sin_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_sequence_length": self.max_sequence_length,
                "pretraining_sequence_length": self.pretraining_sequence_length,
                "inverese_freq_short_factor": self.inverese_freq_short_factor,
                "inverese_freq_long_factor": self.inverese_freq_long_factor,
            }
        )
        return config
