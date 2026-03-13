import math

import keras
import numpy as np

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nAudioRelativePositionEmbedding(keras.layers.Layer):
    """A layer for learning relative position embeddings for audio sequences.

    This layer implements the relative position embedding mechanism used in the
    audio tower of the Gemma3n model. It computes position-aware attention
    scores by generating a timing signal based on relative positions between
    queries and keys, which is then projected and added to the content-based
    attention logits.

    Args:
        hidden_size: int. The size of the hidden state.
        conf_num_attention_heads: int. The number of attention heads.
        conf_attention_context_left: int. The number of steps to attend to in
            the past, including the current step.
        conf_attention_context_right: int. The number of steps to attend to in
            the future.
    """

    def __init__(
        self,
        hidden_size,
        conf_num_attention_heads,
        conf_attention_context_left,
        conf_attention_context_right,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_context_right = conf_attention_context_right
        self.num_heads = conf_num_attention_heads
        self.channels = hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, conf_attention_context_left - 1)
        self.max_forward = conf_attention_context_right
        self.pos_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            name="pos_proj",
            dtype=self.dtype_policy,
        )
        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales, dtype="float32")
            * -log_timescale_increment
        )
        self.inv_timescales = keras.ops.expand_dims(
            keras.ops.expand_dims(
                keras.ops.convert_to_tensor(inv_timescales, dtype="float32"), 0
            ),
            0,
        )

    def build(self, input_shape):
        if not self.pos_proj.built:
            self.pos_proj.build((None, self.channels))
        super().build(input_shape)

    def _get_timing_signal_1d_pos(self, position, dtype):
        position = keras.ops.cast(
            keras.ops.expand_dims(position, axis=-1), "float32"
        )
        pos_shape = keras.ops.shape(position)
        inv_shape = keras.ops.shape(self.inv_timescales)
        target_shape = (pos_shape[0], pos_shape[1], inv_shape[2])
        position = keras.ops.broadcast_to(position, target_shape)
        inv_timescales = keras.ops.broadcast_to(
            self.inv_timescales, target_shape
        )
        scaled_time = position * inv_timescales
        timing_signal = keras.ops.concatenate(
            [keras.ops.sin(scaled_time), keras.ops.cos(scaled_time)], axis=-1
        )
        return keras.ops.cast(timing_signal, dtype)

    def _relative_shift(
        self,
        term_bd_before_shift,
        batch_size,
        num_heads,
        num_query_blocks,
        query_block_size,
        key_context_size,
        max_span_plus_1,
    ):
        msp1_val = max_span_plus_1
        kcs_val = key_context_size
        if not isinstance(msp1_val, int) and hasattr(msp1_val, "shape"):
            msp1_val = keras.ops.shape(msp1_val)[-1]
        if not isinstance(kcs_val, int) and hasattr(kcs_val, "shape"):
            kcs_val = keras.ops.shape(kcs_val)[-1]
        pad_amount_last_dim = (kcs_val + 1) - msp1_val
        padding_tuple = [[0, 0]] * (
            len(keras.ops.shape(term_bd_before_shift)) - 1
        ) + [[0, pad_amount_last_dim]]
        term_bd_padded = keras.ops.pad(term_bd_before_shift, padding_tuple)
        shape_padded = keras.ops.shape(term_bd_padded)
        B = shape_padded[0]
        H = shape_padded[1]
        U = shape_padded[2]
        W = shape_padded[3]
        C_plus_1 = shape_padded[4]
        target_shape_1_last_dim = -1
        if W is not None and C_plus_1 is not None:
            try:
                target_shape_1_last_dim = W * C_plus_1
            except TypeError:
                target_shape_1_last_dim = -1
        term_bd_reshaped = keras.ops.reshape(
            term_bd_padded,
            (
                B if B is not None else -1,
                H if H is not None else -1,
                U if U is not None else -1,
                target_shape_1_last_dim,
            ),
        )
        slice_end = None
        qbs_val = query_block_size
        if not isinstance(qbs_val, int) and hasattr(qbs_val, "shape"):
            qbs_val = keras.ops.shape(qbs_val)[0]
        if qbs_val is not None and kcs_val is not None:
            try:
                slice_end = qbs_val * kcs_val
            except TypeError:
                slice_end = None
        term_bd_reshaped = term_bd_reshaped[..., :slice_end]
        term_bd_shifted = keras.ops.reshape(
            term_bd_reshaped,
            (
                B if B is not None else -1,
                H if H is not None else -1,
                U if U is not None else -1,
                W if W is not None else -1,
                kcs_val if kcs_val is not None else -1,
            ),
        )
        return term_bd_shifted

    def _int8_call(self, queries, keys):
        original_dtype = queries.dtype
        queries_calc = keras.ops.cast(queries, "float32")
        keys_calc = keras.ops.cast(keys, "float32")
        result_calc = self.call(queries_calc, keys_calc)
        return keras.ops.cast(result_calc, original_dtype)

    def call(self, queries, keys):
        batch_size = keras.ops.shape(queries)[0]
        (
            _,
            num_query_blocks,
            query_block_size,
            num_heads,
            head_dim,
        ) = queries.shape
        _, _, key_context_size, _, _ = keys.shape
        pos_indices = keras.ops.expand_dims(
            keras.ops.arange(
                self.max_backward, -self.max_forward - 1, -1, dtype="float32"
            ),
            0,
        )
        max_span_plus_1 = keras.ops.shape(pos_indices)[1]
        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        sin_emb = keras.ops.squeeze(
            keras.ops.reshape(
                projected_sin_emb,
                (1, max_span_plus_1, self.num_heads, self.head_dim),
            ),
            axis=0,
        )
        queries_p = keras.ops.transpose(queries, (0, 3, 1, 2, 4))
        keys_p_t = keras.ops.transpose(keys, (0, 3, 1, 4, 2))
        term_ac = keras.ops.matmul(queries_p, keys_p_t)
        q_permuted = keras.ops.transpose(queries, (0, 3, 1, 2, 4))
        s_permuted = keras.ops.transpose(sin_emb, (1, 2, 0))
        term_bd_unshifed = keras.ops.einsum(
            "bhuwd,hdf->bhuwf", q_permuted, s_permuted
        )
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )
        return term_ac + term_bd_shifted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "conf_num_attention_heads": self.conf_num_attention_heads,
                "conf_attention_context_left": self.conf_attention_context_left,
                "conf_attention_context_right": self.conf_attention_context_right,  # noqa: E501
            }
        )
        return config


class Gemma3nTextAttention(keras.layers.Layer):
    """A multi-head attention layer for text sequences.

    This layer implements the text attention mechanism for the Gemma3n model,
    which is a standard multi-head attention architecture. It includes features
    such as Grouped-Query Attention (GQA), RMS Normalization for query and key
    states, and Rotary Position Embeddings (RoPE) to incorporate positional
    information.

    Args:
        hidden_size: int. The size of the hidden state.
        num_attention_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key and value attention heads.
            If `num_key_value_heads` is not equal to `num_attention_heads`, this
            layer implements Grouped-Query Attention.
        head_dim: int. The dimension of each attention head.
        attention_dropout: float. Dropout probability for the attention scores.
        attention_bias: bool. If `True`, dense layers for query, key, value,
            and output projections will use a bias term.
        rms_norm_eps: float. The epsilon value for RMS Normalization layers.
        rope_max_wavelength: int. The maximum wavelength for the
            rotary position embedding. Defaults to 10000.
        rope_scaling_factor: float. The scaling factor for the
            rotary position embedding. Defaults to 1.0.
        sliding_window: int, optional. The size of the sliding window for
            local attention. If `None`, global attention is used. Defaults to
            `None`.
        is_kv_shared_layer: bool. Whether this layer reuses kv states from a
            previous layer.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        attention_dropout,
        attention_bias,
        rms_norm_eps,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        sliding_window=None,
        is_kv_shared_layer=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.sliding_window = sliding_window
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.is_kv_shared_layer = is_kv_shared_layer
        self.q_proj = keras.layers.Dense(
            self.num_attention_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="q_proj",
            dtype=self.dtype_policy,
        )
        self.k_proj = keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="k_proj",
            dtype=self.dtype_policy,
        )
        self.v_proj = keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="v_proj",
            dtype=self.dtype_policy,
        )
        self.o_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=self.attention_bias,
            name="o_proj",
            dtype=self.dtype_policy,
        )
        self.q_norm = Gemma3nRMSNorm(
            dim=self.head_dim,
            eps=self.rms_norm_eps,
            name="q_norm",
            dtype=self.dtype_policy,
        )
        self.k_norm = Gemma3nRMSNorm(
            dim=self.head_dim,
            eps=self.rms_norm_eps,
            name="k_norm",
            dtype=self.dtype_policy,
        )
        self.v_norm = Gemma3nRMSNorm(
            dim=self.head_dim,
            eps=self.rms_norm_eps,
            with_scale=False,
            name="v_norm",
            dtype=self.dtype_policy,
        )
        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=rope_max_wavelength,
            scaling_factor=rope_scaling_factor or 1.0,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build(
            input_shape[:-1] + (self.num_attention_heads * self.head_dim,)
        )
        norm_shape = input_shape[:-1] + (
            self.num_attention_heads,
            self.head_dim,
        )
        self.q_norm.build(norm_shape)
        k_norm_shape = input_shape[:-1] + (
            self.num_key_value_heads,
            self.head_dim,
        )
        self.k_norm.build(k_norm_shape)
        self.v_norm.build(k_norm_shape)
        super().build(input_shape)

    def _mask_sliding_window(
        self,
        attention_mask,
        cache_update_index=0,
    ):
        batch_size, query_len, key_len = keras.ops.shape(attention_mask)

        # Compute the sliding window for square attention.
        all_ones = keras.ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            # TODO: trui/tril has issues with dynamic shape on the tensorflow
            # backend. We should fix, but use `band_part` for now.
            import tensorflow as tf

            band_size = keras.ops.minimum(key_len, self.sliding_window - 1)
            band_size = keras.ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
        else:
            sliding_mask = keras.ops.triu(
                all_ones, -1 * self.sliding_window + 1
            ) * keras.ops.tril(all_ones, self.sliding_window - 1)
        # Slice the window for short queries during generation.
        start = (cache_update_index, 0)
        sliding_mask = keras.ops.slice(
            sliding_mask, start, (query_len, key_len)
        )
        sliding_mask = keras.ops.expand_dims(sliding_mask, 0)
        return keras.ops.logical_and(
            attention_mask, keras.ops.cast(sliding_mask, "bool")
        )

    def repeat_kv(self, hidden_states):
        if self.num_key_value_groups == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = keras.ops.expand_dims(hidden_states, 2)
        hidden_states = keras.ops.repeat(
            hidden_states, self.num_key_value_groups, axis=2
        )
        return keras.ops.reshape(
            hidden_states,
            (
                batch,
                num_key_value_heads * self.num_key_value_groups,
                slen,
                head_dim,
            ),
        )

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        training=False,
    ):
        scaling = 1.0
        key_states = self.repeat_kv(key)
        value_states = self.repeat_kv(value)
        attn_weights = (
            keras.ops.matmul(
                query, keras.ops.transpose(key_states, (0, 1, 3, 2))
            )
            * scaling
        )
        if attention_mask is not None:
            mask = keras.ops.expand_dims(attention_mask, axis=1)
            mask = mask[:, :, :, : key_states.shape[-2]]
            mask = keras.ops.cast(mask, "bool")
            attn_weights = keras.ops.where(mask, attn_weights, -1e9)
        attn_weights = keras.ops.softmax(attn_weights, axis=-1)
        if training:
            attn_weights = keras.layers.Dropout(dropout)(
                attn_weights, training=training
            )
        attn_output = keras.ops.matmul(attn_weights, value_states)
        attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
        return attn_output, attn_weights

    def call(
        self,
        hidden_states,
        attention_mask,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
        training=False,
    ):
        input_shape = keras.ops.shape(hidden_states)[:-1]
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query_states = self.q_proj(hidden_states)
        query_states = keras.ops.reshape(
            query_states,
            input_shape + (self.num_attention_heads, self.head_dim),
        )
        query_states = self.q_norm(query_states)
        query_states = self.rotary_embedding(
            query_states, start_index=start_index
        )
        query_states = keras.ops.transpose(query_states, (0, 2, 1, 3))
        if self.is_kv_shared_layer:
            key_states = cache[:, 0, ...]
            value_states = cache[:, 1, ...]
        elif cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.k_proj(hidden_states)
            key_update = keras.ops.reshape(
                key_update,
                input_shape + (self.num_key_value_heads, self.head_dim),
            )
            key_update = self.k_norm(key_update)
            key_update = self.rotary_embedding(
                key_update, start_index=start_index
            )
            key_update = keras.ops.transpose(key_update, (0, 2, 1, 3))
            value_update = self.v_proj(hidden_states)
            value_update = keras.ops.reshape(
                value_update,
                input_shape + (self.num_key_value_heads, self.head_dim),
            )
            value_update = self.v_norm(value_update)
            value_update = keras.ops.transpose(value_update, (0, 2, 1, 3))
            start = [0, 0, cache_update_index, 0]
            if cache_update_mask is not None:
                cache_update_mask = keras.ops.expand_dims(
                    keras.ops.expand_dims(cache_update_mask, axis=1),
                    axis=-1,
                )
                key_original = keras.ops.slice(
                    key_cache, start, keras.ops.shape(key_update)
                )
                value_original = keras.ops.slice(
                    value_cache, start, keras.ops.shape(value_update)
                )
                key_update = keras.ops.where(
                    cache_update_mask,
                    key_update,
                    key_original,
                )
                value_update = keras.ops.where(
                    cache_update_mask,
                    value_update,
                    value_original,
                )
            key_states = keras.ops.slice_update(key_cache, start, key_update)
            value_states = keras.ops.slice_update(
                value_cache, start, value_update
            )
            cache = keras.ops.stack((key_states, value_states), axis=1)
        else:
            key_states = self.k_proj(hidden_states)
            key_states = keras.ops.reshape(
                key_states,
                input_shape + (self.num_key_value_heads, self.head_dim),
            )
            key_states = self.k_norm(key_states)
            key_states = self.rotary_embedding(
                key_states, start_index=start_index
            )
            key_states = keras.ops.transpose(key_states, (0, 2, 1, 3))
            value_states = self.v_proj(hidden_states)
            value_states = keras.ops.reshape(
                value_states,
                input_shape + (self.num_key_value_heads, self.head_dim),
            )
            value_states = self.v_norm(value_states)
            value_states = keras.ops.transpose(value_states, (0, 2, 1, 3))
            cache = keras.ops.stack((key_states, value_states), axis=1)
        if self.sliding_window is not None and attention_mask is not None:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index,
            )
        attn_output, attn_weights = self._compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if training else 0.0,
            training=training,
        )
        attn_output = keras.ops.reshape(attn_output, input_shape + (-1,))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, cache

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "attention_dropout": self.attention_dropout,
                "attention_bias": self.attention_bias,
                "rms_norm_eps": self.rms_norm_eps,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "sliding_window": self.sliding_window,
                "is_kv_shared_layer": self.is_kv_shared_layer,
            }
        )
        return config


class Gemma3nAudioAttention(keras.layers.Layer):
    """An attention layer specialized for audio sequences.

    This layer implements the attention mechanism for the audio tower of the
    Gemma3n model. It is designed to handle long audio sequences by processing
    the input in fixed-size chunks. For each chunk of queries, it attends to a
    larger context of keys and values, defined by a left (past) and right
    (future) context window. This allows the model to capture local and more
    distant dependencies efficiently.

    Args:
        hidden_size: int. The size of the hidden state.
        conf_num_attention_heads: int. The number of attention heads.
        conf_attention_chunk_size: int. The size of each processing chunk.
        conf_attention_context_right: int. The number of steps to attend to in
            the future.
        conf_attention_context_left: int. The number of steps to attend to in
            the past, including the current step.
        conf_attention_logit_cap: float. The soft cap value to apply to the
            attention logits.
    """

    def __init__(
        self,
        hidden_size,
        conf_num_attention_heads,
        conf_attention_chunk_size,
        conf_attention_context_right,
        conf_attention_context_left,
        conf_attention_logit_cap,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.num_heads = conf_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = conf_attention_chunk_size
        self.max_future_horizon = conf_attention_context_right
        self.max_past_horizon = max(0, conf_attention_context_left - 1)
        self.attention_logits_soft_cap = conf_attention_logit_cap
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.relative_position_embedding = (
            Gemma3nAudioRelativePositionEmbedding(
                hidden_size,
                conf_num_attention_heads,
                conf_attention_context_left,
                conf_attention_context_right,
                name="relative_position_embedding",
                dtype=self.dtype_policy,
            )
        )
        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            name="q_proj",
            dtype=self.dtype_policy,
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            name="k_proj",
            dtype=self.dtype_policy,
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            name="v_proj",
            dtype=self.dtype_policy,
        )
        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / np.log(1 + np.exp(0.0))  # softplus(0) for numpy
        self.q_scale = q_scale * r_softplus_0

        lower_causal_mask = np.tril(
            np.ones((self.context_size, self.chunk_size), dtype=bool), k=0
        ).T
        upper_causal_mask = np.tril(
            np.ones((self.chunk_size, self.context_size), dtype=bool),
            k=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = np.ones(
            (self.chunk_size, self.context_size), dtype=bool
        )
        local_causal_valid_mask = (
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        )
        self.local_causal_valid_mask = keras.ops.convert_to_tensor(
            local_causal_valid_mask
        )
        self.softcap = keras.ops.convert_to_tensor(
            self.attention_logits_soft_cap, dtype="float32"
        )

    def build(self, input_shape):
        self.per_dim_scale = self.add_weight(
            shape=(self.head_dim,),
            initializer="zeros",
            trainable=True,
            name="per_dim_scale",
            dtype=self.dtype_policy.variable_dtype,
        )
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        q_build_shape = (
            None,
            None,
            self.chunk_size,
            self.num_heads,
            self.head_dim,
        )
        k_build_shape = (
            None,
            None,
            self.context_size,
            self.num_heads,
            self.head_dim,
        )
        self.relative_position_embedding.build((q_build_shape, k_build_shape))
        super().build(input_shape)

    def _pad_dim1(self, x, pad_left, pad_right):
        paddings = [[0, 0], [pad_left, pad_right]] + [
            [0, 0] for _ in range(len(keras.ops.shape(x)) - 2)
        ]
        return keras.ops.pad(x, paddings)

    def _convert_to_block(self, hidden_states):
        b, t = keras.ops.shape(hidden_states)[:2]
        tail_shape_list = list(hidden_states.shape[2:])
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size
        padding_len = num_blocks * self.chunk_size - t
        hidden_states = self._pad_dim1(hidden_states, 0, padding_len)
        permute_dims = [b, num_blocks, self.chunk_size] + tail_shape_list
        return keras.ops.reshape(hidden_states, permute_dims)

    def _extract_block_context(self, hidden_states):
        _, t = keras.ops.shape(hidden_states)[:2]
        num_frames = (t + self.chunk_size - 1) // self.chunk_size
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)
        frame_len = self.context_size
        frame_step = self.chunk_size

        start_indices = keras.ops.arange(0, num_frames) * frame_step
        frame_offsets = keras.ops.arange(0, frame_len)
        indices = keras.ops.expand_dims(
            start_indices, axis=1
        ) + keras.ops.expand_dims(frame_offsets, axis=0)
        return keras.ops.take(hidden_states, indices, axis=1)

    def call(self, hidden_states, mask):
        qkv_shape = keras.ops.shape(hidden_states)[:-1] + (
            self.num_heads,
            self.head_dim,
        )
        query_states = keras.ops.reshape(self.q_proj(hidden_states), qkv_shape)
        key_states = keras.ops.reshape(self.k_proj(hidden_states), qkv_shape)
        value_states = keras.ops.reshape(self.v_proj(hidden_states), qkv_shape)
        per_dim_scale_sp = keras.ops.softplus(self.per_dim_scale)
        query_states = query_states * self.q_scale * per_dim_scale_sp
        batch_size, q_time = keras.ops.shape(query_states)[:2]
        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = keras.ops.shape(query_blocks)[1]
        original_valid_mask = keras.ops.logical_not(mask)
        extracted_valid_mask_blocks = self._extract_block_context(
            original_valid_mask
        )
        mask_block_shape = keras.ops.shape(extracted_valid_mask_blocks)
        if len(mask_block_shape) > 3:
            axes_to_squeeze = [
                i
                for i, dim in enumerate(mask_block_shape)
                if i > 0 and i < len(mask_block_shape) - 1 and dim == 1
            ]
            if axes_to_squeeze:
                extracted_valid_mask_blocks = keras.ops.squeeze(
                    extracted_valid_mask_blocks, axis=axes_to_squeeze
                )
        mask_block_shape = keras.ops.shape(extracted_valid_mask_blocks)
        if (
            len(mask_block_shape) == 4
            and mask_block_shape[2] * mask_block_shape[3] == self.context_size
        ):
            extracted_valid_mask_blocks = keras.ops.reshape(
                extracted_valid_mask_blocks,
                (batch_size, num_query_blocks, self.context_size),
            )
        condition_from_input_validity = keras.ops.expand_dims(
            keras.ops.expand_dims(extracted_valid_mask_blocks, 1), -2
        )
        condition_from_causality = keras.ops.expand_dims(
            keras.ops.expand_dims(
                keras.ops.expand_dims(self.local_causal_valid_mask, 0), 0
            ),
            0,
        )
        final_condition_for_where = keras.ops.logical_and(
            condition_from_input_validity,
            keras.ops.cast(condition_from_causality, "bool"),
        )
        logits = self.relative_position_embedding(query_blocks, key_blocks)
        softcap = keras.ops.cast(self.softcap, dtype=logits.dtype)
        logits = logits / softcap
        logits = keras.ops.tanh(logits)
        logits = logits * softcap
        compute_dtype = logits.dtype
        dtype_str = str(compute_dtype)
        if "float16" in dtype_str or "bfloat16" in dtype_str:
            min_val = np.finfo(np.float16).min
        else:
            min_val = np.finfo(np.float32).min
        min_val = keras.ops.convert_to_tensor(min_val, dtype=compute_dtype)
        logits = keras.ops.where(final_condition_for_where, logits, min_val)
        probabilities = keras.ops.softmax(
            keras.ops.cast(logits, "float32"), axis=-1
        )
        probabilities = keras.ops.cast(probabilities, value_blocks.dtype)
        context_vectors = keras.ops.einsum(
            "bnuwc,bucnh->buwnh", probabilities, value_blocks
        )
        context_vectors = keras.ops.reshape(
            context_vectors,
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            ),
        )
        context_vectors = context_vectors[:, :q_time]
        return context_vectors

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "conf_num_attention_heads": self.conf_num_attention_heads,
                "conf_attention_chunk_size": self.conf_attention_chunk_size,
                "conf_attention_context_right": self.conf_attention_context_right,  # noqa: E501
                "conf_attention_context_left": self.conf_attention_context_left,
                "conf_attention_logit_cap": self.conf_attention_logit_cap,
            }
        )
        return config
