from keras import ops

from keras_hub.src.models.qwen.qwen_attention import QwenAttention


class Qwen2VLAttention(QwenAttention):
    """Qwen2 attention with optional multimodal RoPE (M-RoPE) support.

    Identical to ``QwenAttention``, but when ``position_ids`` is provided the
    rotary embedding frequencies are computed per (temporal, height, width)
    channel and recombined into contiguous sections — matching HuggingFace
    ``Qwen2VLRotaryEmbedding.recomposition_frequencies``.

    Note this is the *contiguous-section* M-RoPE used by Qwen2-VL, not the
    stride-3 interleaved variant used by Qwen3-VL / Qwen3.5.

    Args:
        mrope_section: list of int or None. ``[s_t, s_h, s_w]`` — section
            sizes (in rotary *pairs*) assigned to the temporal, height and
            width channels. Must sum to ``head_dim // 2``. ``None`` disables
            M-RoPE (plain 1-D RoPE).
    """

    def __init__(self, *args, mrope_section=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = mrope_section

    def build(self, inputs_shape):
        super().build(inputs_shape)
        self.head_dim = inputs_shape[-1] // self.num_query_heads
        if self.mrope_section is not None:
            if sum(self.mrope_section) != self.head_dim // 2:
                raise ValueError(
                    f"`mrope_section` must sum to head_dim // 2 "
                    f"({self.head_dim // 2}), got {self.mrope_section} "
                    f"(sum={sum(self.mrope_section)})."
                )

    def _apply_rope(self, x, start_index, position_ids):
        if position_ids is not None and self.mrope_section is not None:
            return self._apply_mrope(x, position_ids)
        return self.rotary_embedding_layer(x, start_index=start_index)

    def _apply_mrope(self, x, position_ids):
        """Apply contiguous-section M-RoPE.

        Args:
            x: Tensor ``(batch, seq_len, num_heads, head_dim)``.
            position_ids: int tensor ``(batch, 3, seq_len)`` — channels are
                ``[temporal, height, width]``.

        Returns:
            Tensor of same shape as ``x`` with M-RoPE applied.
        """
        half = self.head_dim // 2
        idx = ops.arange(0, self.head_dim, 2, dtype="float32")
        inv_freq = ops.power(
            ops.cast(self.rope_max_wavelength, "float32"),
            -idx / ops.cast(self.head_dim, "float32"),
        )  # (half,)

        freqs = []
        for channel in range(3):
            pos = ops.cast(position_ids[:, channel, :], "float32")  # (B,S)
            freqs.append(ops.einsum("bi,j->bij", pos, inv_freq))  # (B,S,half)

        s_t, s_h, s_w = self.mrope_section
        # Recompose contiguous sections: [0:s_t] <- temporal,
        # [s_t:s_t+s_h] <- height, [s_t+s_h:] <- width.
        emb = ops.concatenate(
            [
                freqs[0][..., :s_t],
                freqs[1][..., s_t : s_t + s_h],
                freqs[2][..., s_t + s_h :],
            ],
            axis=-1,
        )  # (B, S, half)
        emb = ops.concatenate([emb, emb], axis=-1)  # (B, S, head_dim)

        cos = ops.expand_dims(ops.cos(emb), axis=2)  # (B, S, 1, head_dim)
        sin = ops.expand_dims(ops.sin(emb), axis=2)

        x_dtype = x.dtype
        x = ops.cast(x, "float32")
        x1, x2 = x[..., :half], x[..., half:]
        rotated = ops.concatenate([-x2, x1], axis=-1)
        return ops.cast(x * cos + rotated * sin, x_dtype)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        position_ids=None,
        training=None,
    ):
        """Forward pass for attention.

        Same signature as ``QwenAttention.call`` plus:

        Args:
            position_ids: Optional int tensor ``(batch, 3, seq_len)`` with
                per-channel M-RoPE positions for the *current* tokens (a
                slice during cached decoding). When provided and
                ``mrope_section`` is set, M-RoPE is applied instead of
                standard 1-D RoPE.
        """
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self._query_dense(hidden_states)
        query = self._apply_rope(query, start_index, position_ids)

        def _compute_key_value(x):
            key, value = self._key_dense(x), self._value_dense(x)
            key = self._apply_rope(key, start_index, position_ids)
            return key, value

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update, value_update = _compute_key_value(hidden_states)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key, value = _compute_key_value(hidden_states)

        # [batch_shape, seq_len, num_key_value_heads, head_dim]
        # -> [batch_shape, seq_len, num_heads, head_dim]
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            cache_update_index=cache_update_index,
        )

        attention_output = self._dropout_layer(
            attention_output, training=training
        )

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update({"mrope_section": self.mrope_section})
        return config
