# Audio Encoder Components
import math
import keras
from keras import ops

from keras_hub.src.models.gemma3n.gemma3n_layer_norm import Gemma3nRMSNorm


class Gemma3nAudioRelativePositionEmbedding(keras.layers.Layer):
    """Relative Position Embedding for the Audio Conformer block."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_heads = config.conf_num_attention_heads
        self.channels = config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, config.conf_attention_context_left - 1)
        self.max_forward = config.conf_attention_context_right

        self.pos_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="pos_proj"
        )

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        self.inv_timescales = min_timescale * ops.exp(
            ops.arange(num_timescales, dtype="float32")
            * -log_timescale_increment
        )

    def _get_timing_signal_1d_pos(self, position):
        scaled_time = ops.expand_dims(position, axis=-1) * ops.expand_dims(
            self.inv_timescales, axis=0
        )
        return ops.concatenate(
            [ops.sin(scaled_time), ops.cos(scaled_time)], axis=-1
        )

    def _relative_shift(
        self,
        x,
        batch_size,
        num_heads,
        num_query_blocks,
        query_block_size,
        key_context_size,
    ):
        # x shape: [B, N, U, W, F_span]
        # Target shape: [B, N, U, W, C]
        f_span = ops.shape(x)[-1]
        pad_amount = key_context_size + 1 - f_span
        padded_x = ops.pad(x, [[0, 0], [0, 0], [0, 0], [0, 0], [0, pad_amount]])
        reshaped_x = ops.reshape(
            padded_x,
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            ),
        )
        sliced_x = reshaped_x[:, :, :, : query_block_size * key_context_size]
        return ops.reshape(
            sliced_x,
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            ),
        )

    def call(self, queries, keys):
        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = (
            ops.shape(queries)
        )
        _, _, key_context_size, _, _ = ops.shape(keys)

        pos_indices = ops.arange(
            self.max_backward, -self.max_forward - 1, -1, dtype="float32"
        )
        max_span_plus_1 = ops.shape(pos_indices)[0]

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(pos_indices)
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        sin_emb = ops.reshape(
            projected_sin_emb, (max_span_plus_1, self.num_heads, self.head_dim)
        )

        queries_p = ops.transpose(queries, [0, 3, 1, 2, 4])
        keys_p_t = ops.transpose(keys, [0, 3, 1, 4, 2])
        term_ac = ops.matmul(queries_p, keys_p_t)

        term_bd_unshifed = ops.einsum("buwnh,fnh->bnuwf", queries, sin_emb)
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        )
        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(keras.layers.Layer):
    """Attention mechanism for the Audio Conformer block."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_heads = config.conf_num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = config.conf_attention_chunk_size
        self.max_future_horizon = config.conf_attention_context_right
        self.max_past_horizon = max(0, config.conf_attention_context_left - 1)
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.attention_logits_soft_cap = config.conf_attention_logit_cap

        self.relative_position_embedding = (
            Gemma3nAudioRelativePositionEmbedding(
                config, name="relative_position_embedding"
            )
        )
        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="v_proj"
        )

    def build(self, input_shape):
        self.per_dim_scale = self.add_weight(
            shape=(self.head_dim,),
            initializer="zeros",
            trainable=True,
            name="per_dim_scale",
        )
        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / math.log(2.0)  # softplus(0) = log(2)
        self.q_scale = q_scale * r_softplus_0

        lower_causal_mask = ops.cast(
            ops.tril(
                ops.ones((self.context_size, self.chunk_size)),
                k=0,
            ),
            "bool",
        )
        upper_causal_mask = ops.cast(
            ops.tril(
                ops.ones((self.chunk_size, self.context_size)),
                k=self.max_past_horizon + self.max_future_horizon,
            ),
            "bool",
        )
        self.local_causal_valid_mask = (
            ops.transpose(lower_causal_mask) & upper_causal_mask
        )
        self.built = True

    def _pad_dim1(self, x, pad_left, pad_right):
        paddings = [[0, 0]] * x.ndim
        paddings[1] = [pad_left, pad_right]
        return ops.pad(x, paddings)

    def _convert_to_block(self, x):
        b, t = ops.shape(x)[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size
        padding_len = num_blocks * self.chunk_size - t
        if padding_len > 0:
            x = self._pad_dim1(x, 0, padding_len)
        return ops.reshape(
            x, (b, num_blocks, self.chunk_size) + ops.shape(x)[2:]
        )

    def _extract_block_context(self, x):
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x_padded = self._pad_dim1(x, pad_left, pad_right)

        # This is equivalent to signal.frame in JAX/SciPy
        # Keras doesn't have a direct equivalent, so we implement it manually
        b, t_padded = ops.shape(x_padded)[:2]
        tail_shape = ops.shape(x_padded)[2:]
        num_blocks = (t_padded - self.context_size) // self.chunk_size + 1

        contexts = []
        for i in range(num_blocks):
            start = i * self.chunk_size
            end = start + self.context_size
            contexts.append(x_padded[:, start:end, ...])

        return ops.stack(contexts, axis=1)

    def call(self, hidden_states, mask):
        qkv_shape = ops.shape(hidden_states)[:-1] + (
            self.num_heads,
            self.head_dim,
        )
        query_states = ops.reshape(self.q_proj(hidden_states), qkv_shape)
        key_states = ops.reshape(self.k_proj(hidden_states), qkv_shape)
        value_states = ops.reshape(self.v_proj(hidden_states), qkv_shape)

        per_dim_scale_sp = ops.softplus(self.per_dim_scale)
        query_states = query_states * self.q_scale * per_dim_scale_sp

        batch_size, q_time = ops.shape(query_states)[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = ops.shape(query_blocks)[1]

        original_valid_mask = ~mask
        extracted_valid_mask_blocks = self._extract_block_context(
            original_valid_mask
        )

        condition_from_input_validity = ops.expand_dims(
            ops.expand_dims(extracted_valid_mask_blocks, 1), -2
        )
        condition_from_causality = ops.expand_dims(
            ops.expand_dims(
                ops.expand_dims(self.local_causal_valid_mask, 0), 0
            ),
            0,
        )
        final_condition = ops.logical_and(
            condition_from_input_validity, condition_from_causality
        )

        logits = self.relative_position_embedding(query_blocks, key_blocks)
        logits = logits / self.attention_logits_soft_cap
        logits = ops.tanh(logits) * self.attention_logits_soft_cap

        logits = ops.where(
            final_condition, logits, ops.cast(float("-inf"), logits.dtype)
        )
        probabilities = ops.softmax(logits, axis=-1)

        context_vectors = ops.einsum(
            "bnuwc,bucnh->buwnh", probabilities, value_blocks
        )
        context_vectors = ops.reshape(
            context_vectors,
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            ),
        )
        return context_vectors[:, :q_time, :, :]


class Gemma3nAudioConformerAttention(keras.layers.Layer):
    """Attention block for the Audio Conformer."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gradient_clipping = config.gradient_clipping
        self.pre_attn_norm = Gemma3nRMSNorm(
            config.hidden_size, name="pre_attn_norm"
        )
        self.attn = Gemma3nAudioAttention(config, name="attn")
        self.post = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="post"
        )
        self.post_norm = Gemma3nRMSNorm(config.hidden_size, name="post_norm")

    def call(self, audio_encodings, audio_mel_mask):
        residual = audio_encodings
        audio_encodings = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        audio_encodings_attn_out = self.attn(
            audio_encodings_norm, audio_mel_mask
        )

        b, t, n, h = ops.shape(audio_encodings_attn_out)
        audio_encodings_reshaped = ops.reshape(
            audio_encodings_attn_out, (b, t, n * h)
        )

        audio_encodings = self.post(audio_encodings_reshaped)
        audio_encodings = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return residual + self.post_norm(audio_encodings)


class Gemma3nAudioConformerFeedForward(keras.layers.Layer):
    """FeedForward block for the Audio Conformer."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.conf_residual_weight

        self.pre_layer_norm = Gemma3nRMSNorm(
            config.hidden_size, name="pre_layer_norm"
        )
        self.ffw_layer_1 = keras.layers.Dense(
            config.hidden_size * 4, use_bias=False, name="ffw_layer_1"
        )
        self.ffw_layer_2 = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="ffw_layer_2"
        )
        self.post_layer_norm = Gemma3nRMSNorm(
            config.hidden_size, name="post_layer_norm"
        )

    def call(self, audio_encodings):
        residual = audio_encodings
        audio_encodings = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.ffw_layer_1(audio_encodings)
        audio_encodings = ops.silu(audio_encodings)
        audio_encodings = self.ffw_layer_2(audio_encodings)
        audio_encodings = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(keras.layers.Layer):
    """Lightweight 1D convolution block for the Audio Conformer."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gradient_clipping = config.gradient_clipping
        self.causal_padding = config.conf_conv_kernel_size - 1

        self.pre_layer_norm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="pre_layer_norm",
        )
        self.linear_start = keras.layers.Dense(
            config.hidden_size * 2, use_bias=False, name="linear_start"
        )
        self.depthwise_conv1d = keras.layers.DepthwiseConv1D(
            kernel_size=config.conf_conv_kernel_size,
            strides=1,
            padding="valid",  # Manual causal padding
            use_bias=False,
            name="depthwise_conv1d",
        )
        self.conv_norm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, name="conv_norm"
        )
        self.linear_end = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="linear_end"
        )

    def call(self, audio_encodings):
        residual = audio_encodings
        x = self.pre_layer_norm(audio_encodings)
        x = self.linear_start(x)
        x = ops.glu(x, axis=-1)

        # Keras DepthwiseConv1D expects (batch, length, channels)
        x_padded = ops.pad(x, [[0, 0], [self.causal_padding, 0], [0, 0]])
        x = self.depthwise_conv1d(x_padded)

        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = ops.silu(x)
        x = self.linear_end(x)
        return x + residual


class Gemma3nAudioConformerBlock(keras.layers.Layer):
    """A single Conformer block for the Audio Encoder."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gradient_clipping = config.gradient_clipping

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(
            config, name="ffw_layer_start"
        )
        self.attention = Gemma3nAudioConformerAttention(
            config, name="attention"
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(config, name="lconv1d")
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(
            config, name="ffw_layer_end"
        )
        self.norm = Gemma3nRMSNorm(config.hidden_size, name="norm")

    def call(self, audio_encodings, audio_mel_mask):
        x = self.ffw_layer_start(audio_encodings)
        x = self.attention(x, audio_mel_mask)
        validity_mask = ops.expand_dims(~audio_mel_mask, axis=-1)
        x_masked = x * ops.cast(validity_mask, x.dtype)
        x = self.lconv1d(x_masked)
        x = self.ffw_layer_end(x)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm(x)


class Gemma3nAudioEncoder(keras.Model):
    """Gemma3n Audio Encoder based on the Universal Speech Model."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # TODO: Implement the audio encoder components
        # This is a placeholder for the full audio encoder implementation.
        # The actual implementation would involve translating:
        # - Gemma3nAudioSubSampleConvProjection
        # - The loop of Gemma3nAudioConformerBlock
        # - The final reduction and masking
        self.config = config
        self.input_layer = keras.layers.Input(
            shape=(None, config.input_feat_size), name="audio_mel"
        )
        self.output_layer = keras.layers.Dense(config.hidden_size)
        self.output_mask = keras.layers.Lambda(lambda x: x)  # Placeholder

    def call(self, audio_mel, audio_mel_mask):
        # Placeholder logic
        x = self.output_layer(self.input_layer)
        return x, self.output_mask(audio_mel_mask)

    def get_config(self):
        # Required for serialization
        return {"config": self.config}