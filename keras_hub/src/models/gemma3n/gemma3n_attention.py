import math

import keras
import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_utils import apply_rotary_pos_emb
from keras_hub.src.models.gemma3n.gemma3n_utils import eager_attention_forward
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
                keras.ops.convert_to_tensor(inv_timescales), 0
            ),
            0,
        )

    def build(self, input_shape):
        self.pos_proj.build((None, self.channels))
        super().build(input_shape)

    def _get_timing_signal_1d_pos(self, position, dtype):
        position = keras.ops.cast(
            keras.ops.expand_dims(position, axis=-1), "float32"
        )
        scaled_time = position * keras.ops.cast(self.inv_timescales, "float32")
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
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1
        padding_tuple = [[0, 0]] * (len(term_bd_before_shift.shape) - 1) + [
            [0, pad_amount_last_dim]
        ]
        term_bd_padded = keras.ops.pad(term_bd_before_shift, padding_tuple)
        term_bd_reshaped = keras.ops.reshape(
            term_bd_padded,
            (
                batch_size,
                num_heads,
                -1,
            ),
        )[:, :, : query_block_size * key_context_size]
        term_bd_shifted = keras.ops.reshape(
            term_bd_reshaped,
            (
                batch_size,
                num_heads,
                -1,
                query_block_size,
                key_context_size,
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
        max_span_plus_1 = pos_indices.shape[1]
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

        q_reshaped_dim = -1
        if num_query_blocks is not None:
            q_reshaped_dim = num_query_blocks * query_block_size

        q_reshaped = keras.ops.reshape(
            q_permuted,
            (
                batch_size * num_heads,
                q_reshaped_dim,
                head_dim,
            ),
        )
        term_bd_unshifed_matmul = keras.ops.matmul(q_reshaped, s_permuted)
        term_bd_unshifed = keras.ops.reshape(
            term_bd_unshifed_matmul,
            (
                batch_size,
                num_heads,
                -1,
                query_block_size,
                max_span_plus_1,
            ),
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
        sliding_window: int, optional. The size of the sliding window for
            local attention. If `None`, global attention is used. Defaults to
            `None`.
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
        sliding_window=None,
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
        self.sliding_window = sliding_window
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
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
        self.k_norm.build(norm_shape)
        self.v_norm.build(norm_shape)
        super().build(input_shape)

    def call(
        self, hidden_states, position_embeddings, attention_mask, training=False
    ):
        input_shape = keras.ops.shape(hidden_states)[:-1]
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states)
        query_states = keras.ops.reshape(
            query_states,
            input_shape + (self.num_attention_heads, self.head_dim),
        )
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(
            query_states, cos, sin, unsqueeze_dim=2
        )
        query_states = keras.ops.transpose(query_states, (0, 2, 1, 3))
        key_states = self.k_proj(hidden_states)
        key_states = keras.ops.reshape(
            key_states, input_shape + (self.num_key_value_heads, self.head_dim)
        )
        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = keras.ops.transpose(key_states, (0, 2, 1, 3))
        value_states = self.v_proj(hidden_states)
        value_states = keras.ops.reshape(
            value_states,
            input_shape + (self.num_key_value_heads, self.head_dim),
        )
        value_states = self.v_norm(value_states)
        value_states = keras.ops.transpose(value_states, (0, 2, 1, 3))
        attn_output, attn_weights = eager_attention_forward(
            query_states,
            key_states,
            value_states,
            self.num_key_value_groups,
            self.head_dim,
            attention_mask,
            dropout=self.attention_dropout if training else 0.0,
            training=training,
        )
        attn_output = keras.ops.reshape(attn_output, input_shape + (-1,))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

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
                "sliding_window": self.sliding_window,
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
        self.relative_position_embedding.build(input_shape)
        super().build(input_shape)

    def _pad_dim1(self, x, pad_left, pad_right):
        paddings = [[0, 0], [pad_left, pad_right]] + [
            [0, 0] for _ in range(len(x.shape) - 2)
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
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)
        _, t = keras.ops.shape(hidden_states)[:2]
        frame_len = self.context_size
        frame_step = self.chunk_size
        num_frames = (t - frame_len) // frame_step + 1

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
        if (
            len(extracted_valid_mask_blocks.shape) == 4
            and extracted_valid_mask_blocks.shape[2]
            * extracted_valid_mask_blocks.shape[3]
            == self.context_size
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
        min_val = np.finfo(keras.backend.floatx()).min
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
