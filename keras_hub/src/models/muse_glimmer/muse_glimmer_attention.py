import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class MuseGlimmerTextAttention(keras.layers.Layer):
    """Grouped-query attention used by MuseGlimmer's text decoder.

    Deviates from a standard GQA block in three ways (see
    `modeling_muse_glimmer.py`, `MuseGlimmerTextAttention.forward`):
    - Query and key each pass through a scaleless RMSNorm after projection;
      when `enable_qk_scale_and_gate=True`, query is then further
      multiplied by `qk_scale_factor`, on top of the standard
      `1/sqrt(head_dim)` scaling applied later in the softmax.
    - RoPE is applied only when `use_rope=True` for this layer; NoPE layers
      (`use_rope=False`) receive no positional signal on Q/K at all.
    - When `enable_qk_scale_and_gate=True`, the attention output is gated
      by `sigmoid(gate_dense(hidden_states))` — the *pre-attention* normed
      input, not the attention output itself — applied after the
      head-concat reshape and before `o_proj`.

    Optionally accepts `context_hidden_states` in `call()` (assistant/
    drafter configuration only, see `MuseGlimmerAssistantAttention.forward`
    in `modeling_muse_glimmer_assistant.py`): when provided, query is
    projected from `hidden_states` only, while key/value are projected
    from `concatenate([context_hidden_states, hidden_states])`. RoPE is
    then applied so query lands on the trailing positions of the combined
    range (`cos[..., -q_len:, :]` in the HF source) while key spans the
    full combined range.

    When `context_hidden_states` and `cache` are both provided (DFlash
    speculative decoding via `MuseGlimmerAssistantCausalLM.call_with_cache`),
    `cache` persists ONLY the context stream — one real target position
    written per drafting cycle at `cache_update_index` — never the noise
    block, since the block is discarded/replaced every cycle regardless
    of accept/reject. This lets later cycles attend back to every
    previously-accepted context position (the real DFlash caching
    benefit: context K/V is computed once and reused, not recomputed
    every cycle) while the block itself is always freshly computed.

    Args:
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads (GQA).
        head_dim: int. Per-head dimension.
        qk_scale_factor: float. Extra multiplier on Q after QK-norm, used
            only when `enable_qk_scale_and_gate=True`.
        rms_norm_eps: float. Epsilon for the QK-norm.
        use_rope: bool. Whether this layer applies rotary position
            embeddings to Q/K.
        rope_max_wavelength: float. RoPE base theta.
        sliding_window_size: int or None. Sliding window size, or `None`
            for full attention.
        enable_qk_scale_and_gate: bool. Whether to apply the extra
            `qk_scale_factor` multiply on Q and the sigmoid output gate.
            Defaults to `True`. The assistant/drafter configuration sets
            this to `False`, since `MuseGlimmerAssistantAttention` has
            neither.
        kernel_initializer: initializer for the dense projections.
        dropout: float. Attention dropout rate.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        qk_scale_factor=1.0,
        rms_norm_eps=1e-6,
        use_rope=True,
        rope_max_wavelength=500000.0,
        sliding_window_size=None,
        enable_qk_scale_and_gate=True,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_scale_factor = qk_scale_factor
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.rope_max_wavelength = rope_max_wavelength
        self.sliding_window_size = sliding_window_size
        self.enable_qk_scale_and_gate = enable_qk_scale_and_gate
        self.dropout = dropout
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self._query_dense = keras.layers.EinsumDense(
            "bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        self._key_dense = keras.layers.EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._qk_norm = MuseGlimmerRMSNorm(
            eps=self.rms_norm_eps,
            with_scale=False,
            dtype=self.dtype_policy,
            name="qk_norm",
        )

        if self.enable_qk_scale_and_gate:
            # Per-token, head-agnostic sigmoid gate computed from the same
            # normed input used for Q/K/V (not from the attention output).
            self._gate_dense = keras.layers.EinsumDense(
                "bqm,muh->bquh",
                output_shape=(None, self.num_query_heads, self.head_dim),
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="gate",
            )
            self._gate_dense.build(inputs_shape)

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        self._softmax = keras.layers.Softmax(axis=-1, dtype="float32")
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy
        )

        if self.use_rope:
            self.rotary_embedding_layer = RotaryEmbedding(
                max_wavelength=self.rope_max_wavelength,
                dtype=self.dtype_policy,
            )
        self.built = True

    def call(
        self,
        hidden_states,
        context_hidden_states=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self._query_dense(hidden_states)
        query = self._qk_norm(query)
        if self.enable_qk_scale_and_gate:
            query = query * self.qk_scale_factor

        # Assistant/drafter configuration only: K/V see the concatenated
        # context + block window, Q sees only the block window. See
        # `MuseGlimmerAssistantAttention.forward` in
        # `modeling_muse_glimmer_assistant.py`.
        if context_hidden_states is not None:
            kv_input = ops.concatenate(
                [context_hidden_states, hidden_states], axis=1
            )
        else:
            kv_input = hidden_states

        # Length of the injected context prefix, if any — 0 collapses every
        # offset below to the plain single-length case. Distinct from
        # `key_len - query_len` under caching, where the key axis is the
        # padded cache buffer, not `context_len + query_len`.
        context_len = (
            ops.shape(context_hidden_states)[1]
            if context_hidden_states is not None
            else 0
        )

        if self.use_rope:
            # Q gets only the trailing `q_len` rotary positions of the
            # combined K/V range; K spans the full combined range. Matches
            # `apply_rotary_pos_emb`'s `cos[..., -q_len:, :]` (Q) vs. full
            # `cos` (K) slicing in the HF source. Collapses to the plain
            # single-length case when `context_hidden_states` is `None`.
            query = self.rotary_embedding_layer(
                query, start_index=start_index + context_len
            )

        if self.enable_qk_scale_and_gate:
            gate = self._gate_dense(hidden_states)

        def _compute_key_value(x, position_start):
            key = self._key_dense(x)
            key = self._qk_norm(key)
            if self.use_rope:
                key = self.rotary_embedding_layer(
                    key, start_index=position_start
                )
            value = self._value_dense(x)
            return key, value

        # DFlash context caching: `cache` persists only the context
        # stream (see class docstring); the noise block is always fresh.
        is_context_cache = context_hidden_states is not None and (
            cache is not None
        )

        if is_context_cache:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is not None:
                new_context_key, new_context_value = _compute_key_value(
                    context_hidden_states, cache_update_index
                )
                start = [0, cache_update_index, 0, 0]
                key_cache = ops.slice_update(key_cache, start, new_context_key)
                value_cache = ops.slice_update(
                    value_cache, start, new_context_value
                )
            cache = ops.stack((key_cache, value_cache), axis=1)
            block_key, block_value = _compute_key_value(
                hidden_states, start_index + context_len
            )
            key = ops.concatenate([key_cache, block_key], axis=1)
            value = ops.concatenate([value_cache, block_value], axis=1)
        elif cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key, value = key_cache, value_cache
            else:
                key_update, value_update = _compute_key_value(
                    kv_input, start_index
                )
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            key, value = _compute_key_value(kv_input, start_index)

        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training=training,
            cache_update_index=(
                cache_update_index if cache_update_index is not None else 0
            ),
            context_len=context_len,
            # The caller (`MuseGlimmerTextDecoder`) builds a complete,
            # already sliding-window-aware mask for the context-cache
            # case, since the combined context-buffer + fresh-block key
            # axis isn't the simple square window `_mask_sliding_window`
            # assumes. Applying it here would double-mask incorrectly.
            skip_sliding_window_mask=is_context_cache,
        )
        attention_output = self._dropout_layer(
            attention_output, training=training
        )

        if self.enable_qk_scale_and_gate:
            attention_output = attention_output * ops.sigmoid(
                ops.cast(gate, attention_output.dtype)
            )
        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        training,
        cache_update_index,
        context_len=0,
        skip_sliding_window_mask=False,
    ):
        if self.sliding_window_size and not skip_sliding_window_mask:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index,
                context_len=context_len,
            )
        attention_scores = ops.einsum("bquh,bkuh->buqk", query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, attention_scores.dtype),
        )
        if attention_mask is not None:
            attention_scores = self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        else:
            attention_scores = self._softmax(attention_scores)
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_scores = self._dropout_layer(
            attention_scores, training=training
        )
        return ops.einsum("buqk,bkuh->bquh", attention_scores, value)

    def _mask_sliding_window(
        self, attention_mask, cache_update_index=0, context_len=0
    ):
        _, query_len, key_len = ops.shape(attention_mask)
        all_ones = ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            band_size = ops.minimum(key_len, self.sliding_window_size - 1)
            band_size = ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
        else:
            sliding_mask = ops.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * ops.tril(all_ones, self.sliding_window_size - 1)
        # Query occupies the trailing `query_len` rows of the combined
        # range whenever context injection prepends `context_len` extra key
        # positions (no KV cache in that configuration, so
        # `cache_update_index` is always 0 there) — mirrors the RoPE offset
        # applied to Q in `call()`. `context_len` is 0 outside the
        # assistant/drafter configuration, collapsing to the original
        # cache-relative offset.
        start = (cache_update_index + context_len, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "qk_scale_factor": self.qk_scale_factor,
                "rms_norm_eps": self.rms_norm_eps,
                "use_rope": self.use_rope,
                "rope_max_wavelength": self.rope_max_wavelength,
                "sliding_window_size": self.sliding_window_size,
                "enable_qk_scale_and_gate": self.enable_qk_scale_and_gate,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
