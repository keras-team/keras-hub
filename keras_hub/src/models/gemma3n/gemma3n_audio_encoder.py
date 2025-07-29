# Audio Encoder Components
import math

import keras
from keras import ops

from keras_hub.src.models.gemma3n.gemma3n_layer_norm import Gemma3nRMSNorm


class Gemma3nAudioRelativePositionEmbedding(keras.layers.Layer):
    """Relative Position Embedding for the Audio Conformer block."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_context_left,
        attention_context_right,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_backward = max(0, attention_context_left - 1)
        self.max_forward = attention_context_right

    def build(self, input_shape):
        self.pos_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="pos_proj"
        )
        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        self.inv_timescales = self.add_weight(
            shape=(num_timescales,),
            initializer=keras.initializers.Constant(
                min_timescale
                * ops.exp(
                    ops.arange(num_timescales, dtype="float32")
                    * -log_timescale_increment
                )
            ),
            trainable=False,
            name="inv_timescales",
        )
        self.built = True

    def _get_timing_signal_1d_pos(self, position):
        scaled_time = ops.expand_dims(position, axis=-1) * ops.expand_dims(
            self.inv_timescales, axis=0
        )
        return ops.concatenate(
            [ops.sin(scaled_time), ops.cos(scaled_time)], axis=-1
        )

    def _relative_shift(self, x, key_context_size):
        batch_size, num_heads, num_query_blocks, query_block_size, f_span = (
            ops.shape(x)
        )
        pad_amount_last_dim = key_context_size + 1 - f_span
        paddings = [[0, 0]] * (x.ndim - 1) + [[0, pad_amount_last_dim]]
        padded_x = ops.pad(x, paddings)
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
        _, _, _, num_heads, head_dim = ops.shape(queries)
        batch_size, num_query_blocks, key_context_size, _, _ = ops.shape(keys)
        query_block_size = ops.shape(queries)[2]

        pos_indices = ops.arange(
            self.max_backward, -self.max_forward - 1, -1, dtype="float32"
        )
        sin_emb_timing_signal = self._get_timing_signal_1d_pos(pos_indices)
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        sin_emb = ops.reshape(projected_sin_emb, (-1, num_heads, head_dim))

        queries_p = ops.transpose(queries, [0, 3, 1, 2, 4])
        keys_p_t = ops.transpose(keys, [0, 3, 1, 4, 2])
        term_ac = ops.matmul(queries_p, keys_p_t)
        term_bd_unshifed = ops.einsum("buwnh,fnh->bnuwf", queries, sin_emb)
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed, key_context_size
        )
        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(keras.layers.Layer):
    """Attention mechanism for the Audio Conformer block."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_chunk_size,
        attention_context_left,
        attention_context_right,
        attention_logit_cap,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.chunk_size = attention_chunk_size
        self.attention_context_left = attention_context_left
        self.attention_context_right = attention_context_right
        self.attention_logit_cap = attention_logit_cap

        self.head_dim = hidden_size // num_attention_heads
        self.max_future_horizon = attention_context_right
        self.max_past_horizon = max(0, attention_context_left - 1)
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.relative_position_embedding = (
            Gemma3nAudioRelativePositionEmbedding(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_context_left=attention_context_left,
                attention_context_right=attention_context_right,
                name="relative_position_embedding",
            )
        )

    def build(self, input_shape):
        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="v_proj"
        )
        self.per_dim_scale = self.add_weight(
            shape=(self.head_dim,),
            initializer="zeros",
            trainable=True,
            name="per_dim_scale",
        )
        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / math.log(2.0)
        self.q_scale = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(q_scale * r_softplus_0),
            trainable=False,
            name="q_scale",
        )
        lower_causal_mask = ops.tril(
            ops.ones((self.context_size, self.chunk_size)), k=0
        )
        upper_causal_mask = ops.tril(
            ops.ones((self.chunk_size, self.context_size)),
            k=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = ops.logical_and(
            ops.transpose(lower_causal_mask), upper_causal_mask
        )
        self.local_causal_valid_mask = self.add_weight(
            shape=(self.chunk_size, self.context_size),
            initializer=keras.initializers.Constant(
                ops.cast(local_causal_valid_mask, "bool")
            ),
            trainable=False,
            name="local_causal_valid_mask",
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
        pad_right = self.max_future_horizon
        x_padded = self._pad_dim1(x, pad_left, pad_right)
        b, t_padded = ops.shape(x_padded)[:2]
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
        query_states *= self.q_scale * ops.softplus(self.per_dim_scale)
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
            self.local_causal_valid_mask, axis=(0, 1, 2)
        )
        final_condition = ops.logical_and(
            condition_from_input_validity, condition_from_causality
        )
        logits = self.relative_position_embedding(query_blocks, key_blocks)
        logits = (
            ops.tanh(logits / self.attention_logit_cap)
            * self.attention_logit_cap
        )
        logits = ops.where(
            final_condition, logits, ops.cast(-1e30, logits.dtype)
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
    """Wrapper for the main Audio Attention mechanism."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_chunk_size,
        attention_context_left,
        attention_context_right,
        attention_logit_cap,
        gradient_clipping,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_chunk_size = attention_chunk_size
        self.attention_context_left = attention_context_left
        self.attention_context_right = attention_context_right
        self.attention_logit_cap = attention_logit_cap
        self.gradient_clipping = gradient_clipping

    def build(self, input_shape):
        self.pre_attn_norm = Gemma3nRMSNorm(
            self.hidden_size, name="pre_attn_norm"
        )
        self.attn = Gemma3nAudioAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_chunk_size=self.attention_chunk_size,
            attention_context_left=self.attention_context_left,
            attention_context_right=self.attention_context_right,
            attention_logit_cap=self.attention_logit_cap,
            name="attn",
        )
        self.post = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="post"
        )
        self.post_norm = Gemma3nRMSNorm(self.hidden_size, name="post_norm")
        self.built = True

    def call(self, audio_encodings, audio_mel_mask):
        residual = audio_encodings
        audio_encodings_clipped = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings_clipped)
        audio_encodings_attn_out = self.attn(
            audio_encodings_norm, audio_mel_mask
        )
        b, t, n, h = ops.shape(audio_encodings_attn_out)
        audio_encodings_reshaped = ops.reshape(
            audio_encodings_attn_out, (b, t, n * h)
        )
        audio_encodings_post = self.post(audio_encodings_reshaped)
        audio_encodings_post_clipped = ops.clip(
            audio_encodings_post,
            -self.gradient_clipping,
            self.gradient_clipping,
        )
        return residual + self.post_norm(audio_encodings_post_clipped)


class Gemma3nAudioConformerFeedForward(keras.layers.Layer):
    """FeedForward block for the Audio Conformer."""

    def __init__(
        self, hidden_size, gradient_clipping, residual_weight, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.gradient_clipping = gradient_clipping
        self.post_layer_scale = residual_weight

    def build(self, input_shape):
        self.pre_layer_norm = Gemma3nRMSNorm(
            self.hidden_size, name="pre_layer_norm"
        )
        self.ffw_layer_1 = keras.layers.Dense(
            self.hidden_size * 4, use_bias=False, name="ffw_layer_1"
        )
        self.ffw_layer_2 = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="ffw_layer_2"
        )
        self.post_layer_norm = Gemma3nRMSNorm(
            self.hidden_size, name="post_layer_norm"
        )
        self.built = True

    def call(self, audio_encodings):
        residual = audio_encodings
        audio_encodings_clipped = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_layer_norm(audio_encodings_clipped)
        ffw1_out = self.ffw_layer_1(audio_encodings_norm)
        ffw1_activated = ops.silu(ffw1_out)
        ffw2_out = self.ffw_layer_2(ffw1_activated)
        ffw2_clipped = ops.clip(
            ffw2_out, -self.gradient_clipping, self.gradient_clipping
        )
        ffw_normed = self.post_layer_norm(ffw2_clipped)
        return residual + (ffw_normed * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(keras.layers.Layer):
    """Lightweight 1D convolution block for the Audio Conformer."""

    def __init__(
        self, conv_kernel_size, gradient_clipping, rms_norm_eps, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_kernel_size = conv_kernel_size
        self.gradient_clipping = gradient_clipping
        self.rms_norm_eps = rms_norm_eps
        self.causal_padding = self.conv_kernel_size - 1

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.pre_layer_norm = Gemma3nRMSNorm(
            hidden_dim, epsilon=self.rms_norm_eps, name="pre_layer_norm"
        )
        self.linear_start = keras.layers.Dense(
            hidden_dim * 2, use_bias=False, name="linear_start"
        )
        self.depthwise_conv1d = keras.layers.DepthwiseConv1D(
            kernel_size=self.conv_kernel_size,
            strides=1,
            padding="valid",
            use_bias=False,
            name="depthwise_conv1d",
        )
        self.conv_norm = Gemma3nRMSNorm(
            hidden_dim, epsilon=self.rms_norm_eps, name="conv_norm"
        )
        self.linear_end = keras.layers.Dense(
            hidden_dim, use_bias=False, name="linear_end"
        )
        self.built = True

    def call(self, audio_encodings):
        residual = audio_encodings
        x = self.pre_layer_norm(audio_encodings)
        x = self.linear_start(x)
        x = ops.glu(x, axis=-1)
        x_padded = ops.pad(x, [[0, 0], [self.causal_padding, 0], [0, 0]])
        x = self.depthwise_conv1d(x_padded)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = ops.silu(x)
        x = self.linear_end(x)
        return x + residual


class Gemma3nAudioConformerBlock(keras.layers.Layer):
    """A single Conformer block for the Audio Encoder."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_chunk_size,
        attention_context_left,
        attention_context_right,
        attention_logit_cap,
        gradient_clipping,
        residual_weight,
        conv_kernel_size,
        rms_norm_eps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_chunk_size = attention_chunk_size
        self.attention_context_left = attention_context_left
        self.attention_context_right = attention_context_right
        self.attention_logit_cap = attention_logit_cap
        self.gradient_clipping = gradient_clipping
        self.residual_weight = residual_weight
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps

    def build(self, input_shape):
        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(
            hidden_size=self.hidden_size,
            gradient_clipping=self.gradient_clipping,
            residual_weight=self.residual_weight,
            name="ffw_layer_start",
        )
        self.attention = Gemma3nAudioConformerAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_chunk_size=self.attention_chunk_size,
            attention_context_left=self.attention_context_left,
            attention_context_right=self.attention_context_right,
            attention_logit_cap=self.attention_logit_cap,
            gradient_clipping=self.gradient_clipping,
            name="attention",
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(
            conv_kernel_size=self.conv_kernel_size,
            gradient_clipping=self.gradient_clipping,
            rms_norm_eps=self.rms_norm_eps,
            name="lconv1d",
        )
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(
            hidden_size=self.hidden_size,
            gradient_clipping=self.gradient_clipping,
            residual_weight=self.residual_weight,
            name="ffw_layer_end",
        )
        self.norm = Gemma3nRMSNorm(self.hidden_size, name="norm")
        self.built = True

    def call(self, audio_encodings, audio_mel_mask):
        x = self.ffw_layer_start(audio_encodings)
        x = self.attention(x, audio_mel_mask)
        validity_mask = ops.expand_dims(~audio_mel_mask, axis=-1)
        x_masked = x * ops.cast(validity_mask, x.dtype)
        x = self.lconv1d(x_masked)
        x = self.ffw_layer_end(x)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm(x)


class Gemma3nAudioSSCPConvBlock(keras.layers.Layer):
    """A single 2D convolution block for the audio subsampler."""

    def __init__(
        self, in_channels, out_channels, kernel_size, strides, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=False,
            name="conv",
        )
        # NOTE: Using RMSNorm as a substitute for the original CumulativeGroupNorm
        self.norm = Gemma3nRMSNorm(name="norm")
        self.activation = keras.layers.Activation("relu", name="activation")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.activation(x)


class Gemma3nAudioSubSampleConvProjection(keras.layers.Layer):
    """
    Subsamples the input audio features using 2D convolutions and projects them.
    """

    def __init__(self, final_dim=1536, **kwargs):
        super().__init__(**kwargs)
        self.final_dim = final_dim
        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            1, 128, (3, 3), (2, 2), name="conv_0"
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            128, 32, (3, 3), (2, 2), name="conv_1"
        )
        # After two strides of (2,2), the feature dim will be (orig_freq / 4) * 32 channels.
        # The original paper uses 128 mel bins, so (128 / 4) * 32 = 32 * 32 = 1024.
        self.input_proj_linear = keras.layers.Dense(
            self.final_dim, use_bias=False, name="input_proj_linear"
        )

    def call(self, inputs):
        # Input shape: (batch, time, freq)
        # Add a channel dimension to treat it as an image
        x = ops.expand_dims(inputs, axis=-1)  # -> (batch, time, freq, 1)

        x = self.conv_0(x)  # -> (batch, time/2, freq/2, 128)
        x = self.conv_1(x)  # -> (batch, time/4, freq/4, 32)

        # Reshape to (batch, new_time, new_features) for the linear projection
        b, t, f, c = ops.shape(x)
        x = ops.reshape(x, (b, t, f * c))

        x = self.input_proj_linear(x)  # -> (batch, new_time, final_dim)
        return x


class Gemma3nAudioEncoder(keras.layers.Layer):
    """
    The complete Gemma3n audio encoder tower, composed of a subsampler
    and a stack of Conformer blocks.
    """

    def __init__(
        self,
        num_conformer_layers,
        hidden_size,
        num_attention_heads,
        attention_chunk_size,
        attention_context_left,
        attention_context_right,
        attention_logit_cap,
        gradient_clipping,
        residual_weight,
        conv_kernel_size,
        rms_norm_eps,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_conformer_layers = num_conformer_layers

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(
            final_dim=hidden_size, name="subsample_conv_projection"
        )

        self.conformer = []
        for i in range(num_conformer_layers):
            block = Gemma3nAudioConformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_chunk_size=attention_chunk_size,
                attention_context_left=attention_context_left,
                attention_context_right=attention_context_right,
                attention_logit_cap=attention_logit_cap,
                gradient_clipping=gradient_clipping,
                residual_weight=residual_weight,
                conv_kernel_size=conv_kernel_size,
                rms_norm_eps=rms_norm_eps,
                name=f"conformer_block_{i}",
                dtype=dtype,
            )
            self.conformer.append(block)

    def call(self, audio_features, audio_mel_mask=None):
        # 1. Subsample features
        x = self.subsample_conv_projection(audio_features)

        # 2. Subsample the mask if provided.
        # The time dimension is reduced by a factor of 4.
        if audio_mel_mask is not None:
            mask_squeezed = ops.expand_dims(
                ops.cast(audio_mel_mask, "float32"), axis=-1
            )
            # Use max pooling to ensure that if any of the original 4 steps was masked,
            # the new subsampled step is also masked.
            subsampled_mask = keras.layers.MaxPooling1D(
                pool_size=4, strides=4, padding="same"
            )(mask_squeezed)
            subsampled_mask = ops.squeeze(
                ops.cast(subsampled_mask, "bool"), axis=-1
            )
            # Adjust length due to potential padding
            target_len = ops.shape(x)[1]
            subsampled_mask = subsampled_mask[:, :target_len]
        else:
            subsampled_mask = None

        # 3. Pass through Conformer blocks
        for block in self.conformer:
            x = block(x, subsampled_mask)

        return x
