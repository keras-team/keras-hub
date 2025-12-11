import keras

from keras_hub.src.models.gemma3n.gemma3n_attention import Gemma3nAudioAttention
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nAudioCumulativeGroupNorm(keras.layers.Layer):
    """A cumulative group normalization layer for audio features.

    This layer normalizes the input hidden states based on cumulative statistics
    calculated over the time dimension. It is designed to process audio
    spectrograms or similar sequential data.

    Args:
        num_channels: int. The number of channels for normalization.
        feature_dims: tuple. The dimensions of the features to be normalized.
        eps: float. A small epsilon value to add to the variance to avoid
            division by zero.
    """

    def __init__(
        self,
        num_channels,
        feature_dims,
        eps=1e-3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.num_channels,),
            initializer="ones",
            trainable=True,
            name="scale",
            dtype=self.dtype_policy.variable_dtype,
        )
        super().build(input_shape)

    def _int8_call(self, hidden_states):
        original_dtype = hidden_states.dtype
        x_calc = keras.ops.cast(hidden_states, "float32")
        result_calc = self.call(x_calc)
        return keras.ops.cast(result_calc, original_dtype)

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        x_calc = keras.ops.cast(hidden_states, "float32")
        mask_calc = keras.ops.ones_like(x_calc, dtype="float32")
        sum_values_at_t = keras.ops.sum(
            x_calc, axis=self.reduction_axes, keepdims=True
        )
        cum_sum_values = keras.ops.cumsum(sum_values_at_t, axis=1)
        elements_in_group_at_t = keras.ops.sum(
            mask_calc, axis=self.reduction_axes, keepdims=True
        )
        cum_count_elements = keras.ops.cumsum(elements_in_group_at_t, axis=1)
        safe_cum_count_elements = keras.ops.maximum(cum_count_elements, 1.0)
        cum_mean = cum_sum_values / safe_cum_count_elements
        squared_diff_from_mean = keras.ops.square(x_calc - cum_mean)
        sum_sq_diff_at_t = keras.ops.sum(
            squared_diff_from_mean, axis=self.reduction_axes, keepdims=True
        )
        cum_sum_sq_diff = keras.ops.cumsum(sum_sq_diff_at_t, axis=1)
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements
        normalized_x = (x_calc - cum_mean) * keras.ops.rsqrt(
            cum_variance + self.eps
        )
        scale_view_shape = [1] * (len(hidden_states.shape) - 1) + [
            self.num_channels
        ]
        reshaped_scale = keras.ops.reshape(self.scale, scale_view_shape)
        normalized_x = normalized_x * keras.ops.cast(reshaped_scale, "float32")
        final_output = normalized_x * mask_calc
        return keras.ops.cast(final_output, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "feature_dims": self.feature_dims,
                "eps": self.eps,
            }
        )
        return config


class Gemma3nAudioSSCPConvBlock(keras.layers.Layer):
    """A single SSCP (Spectrogram Sub-sampling Convolutional Preprocessor)
    block.

    This block consists of a 2D convolution, a cumulative group normalization
    layer, and a ReLU activation. It is used to process and downsample audio
    spectrograms.

    Args:
        idx: int. The index of the convolutional block.
        input_freq_dim: int. The frequency dimension of the input spectrogram.
        sscp_conv_channel_size: list or tuple. A sequence containing the number
            of output channels for each convolutional block in the SSCP stack.
        sscp_conv_kernel_size: list or tuple. A sequence of kernel sizes for
            each convolutional block.
        sscp_conv_stride_size: list or tuple. A sequence of stride sizes for
            each convolutional block.
        sscp_conv_group_norm_eps: float. The epsilon value for the cumulative
            group normalization layer.
        manual_padding: tuple. A tuple of 4 integers specifying the manual
            padding to be applied as (pad_w_left, pad_w_right, pad_h_top,
            pad_h_bottom).
    """

    def __init__(
        self,
        idx,
        input_freq_dim,
        sscp_conv_channel_size,
        sscp_conv_kernel_size,
        sscp_conv_stride_size,
        sscp_conv_group_norm_eps,
        manual_padding=(0, 0, 0, 0),
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.idx = idx
        self.input_freq_dim = input_freq_dim
        self.sscp_conv_channel_size = sscp_conv_channel_size
        self.sscp_conv_kernel_size = sscp_conv_kernel_size
        self.sscp_conv_stride_size = sscp_conv_stride_size
        self.sscp_conv_group_norm_eps = sscp_conv_group_norm_eps
        self.manual_padding = manual_padding
        out_channels = sscp_conv_channel_size[idx]
        kernel_h, kernel_w = sscp_conv_kernel_size[idx]
        stride_h, stride_w = sscp_conv_stride_size[idx]
        self.conv = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(kernel_h, kernel_w),
            strides=(stride_h, stride_w),
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            name="conv",
            dtype=self.dtype_policy,
        )
        f_in_padded = (
            input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        )
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1
        self.norm = Gemma3nAudioCumulativeGroupNorm(
            num_channels=out_channels,
            feature_dims=(f_out_conv,),
            eps=sscp_conv_group_norm_eps,
            name="norm",
            dtype=self.dtype_policy,
        )
        self.activation = keras.layers.ReLU(
            name="activation", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        _, c_in, h, w = input_shape
        if h is not None:
            padded_h = h + self.manual_padding[2] + self.manual_padding[3]
        else:
            padded_h = None
        padded_w = w + self.manual_padding[0] + self.manual_padding[1]
        conv_input_shape = (None, padded_h, padded_w, c_in)
        if not self.conv.built:
            self.conv.build(conv_input_shape)
        if h is not None:
            h_out = (padded_h - self.conv.kernel_size[0]) // self.conv.strides[
                0
            ] + 1
        else:
            h_out = None
        w_out = (padded_w - self.conv.kernel_size[1]) // self.conv.strides[
            1
        ] + 1
        norm_input_shape = (None, h_out, w_out, self.conv.filters)
        if not self.norm.built:
            self.norm.build(norm_input_shape)
        super().build(input_shape)

    def call(self, audio_encodings):
        audio_encodings_nhwc = keras.ops.transpose(
            audio_encodings, (0, 2, 3, 1)
        )
        keras_padding = [
            [0, 0],
            [self.manual_padding[2], self.manual_padding[3]],
            [self.manual_padding[0], self.manual_padding[1]],
            [0, 0],
        ]
        audio_encodings_padded = keras.ops.pad(
            audio_encodings_nhwc,
            keras_padding,
            mode="constant",
            constant_values=0.0,
        )
        audio_encodings_conv = self.conv(audio_encodings_padded)
        x_normed = self.norm(audio_encodings_conv)
        audio_encodings_normed = keras.ops.transpose(x_normed, (0, 3, 1, 2))
        return self.activation(audio_encodings_normed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "idx": self.idx,
                "input_freq_dim": self.input_freq_dim,
                "sscp_conv_channel_size": self.sscp_conv_channel_size,
                "sscp_conv_kernel_size": self.sscp_conv_kernel_size,
                "sscp_conv_stride_size": self.sscp_conv_stride_size,
                "sscp_conv_group_norm_eps": self.sscp_conv_group_norm_eps,
                "manual_padding": self.manual_padding,
            }
        )
        return config


class Gemma3nAudioConformerFeedForward(keras.layers.Layer):
    """The feed-forward module for the Conformer block.

    This module implements the feed-forward sub-layer of a Conformer block,
    which consists of pre-layer normalization, two dense layers with a SiLU
    activation function in between, post-layer normalization, and a residual
    connection.

    Args:
        hidden_size: int. The hidden size of the input and output tensors.
        gradient_clipping: float. The maximum absolute value for gradient
            clipping.
        conf_residual_weight: float. The weight applied to the output of the
            sub-layer before adding the residual connection.
        rms_norm_eps: float. The epsilon value for the RMS normalization layers.
    """

    def __init__(
        self,
        hidden_size,
        gradient_clipping,
        conf_residual_weight,
        rms_norm_eps,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.gradient_clipping = gradient_clipping
        self.conf_residual_weight = conf_residual_weight
        self.rms_norm_eps = rms_norm_eps
        self.pre_layer_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="pre_layer_norm",
            dtype=self.dtype_policy,
        )
        self.ffw_layer_1 = keras.layers.Dense(
            hidden_size * 4,
            use_bias=False,
            name="ffw_layer_1",
            dtype=self.dtype_policy,
        )
        self.ffw_layer_2 = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="ffw_layer_2",
            dtype=self.dtype_policy,
        )
        self.post_layer_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="post_layer_norm",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.pre_layer_norm.build(input_shape)
        self.ffw_layer_1.build(input_shape)
        ffw1_output_shape = input_shape[:-1] + (self.hidden_size * 4,)
        self.ffw_layer_2.build(ffw1_output_shape)
        self.post_layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, audio_encodings):
        residual = audio_encodings
        audio_encodings = keras.ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.ffw_layer_1(audio_encodings)
        audio_encodings = keras.activations.silu(audio_encodings)
        audio_encodings = self.ffw_layer_2(audio_encodings)
        audio_encodings = keras.ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.conf_residual_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "gradient_clipping": self.gradient_clipping,
                "conf_residual_weight": self.conf_residual_weight,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


class Gemma3nAudioConformerLightConv1d(keras.layers.Layer):
    """The lightweight 1D convolution module for the Conformer block.

    This module implements the convolution sub-layer of a Conformer block,
    which consists of pre-layer normalization, a gated linear unit (GLU), a
    lightweight depthwise 1D convolution, and a final projection, followed by a
    residual connection.

    Args:
        hidden_size: int. The hidden size of the input and output tensors.
        rms_norm_eps: float. The epsilon value for the RMS normalization layers.
        conf_conv_kernel_size: int. The kernel size for the depthwise 1D
            convolution.
        gradient_clipping: float. The maximum absolute value for gradient
            clipping.
    """

    def __init__(
        self,
        hidden_size,
        rms_norm_eps,
        conf_conv_kernel_size,
        gradient_clipping,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.conf_conv_kernel_size = conf_conv_kernel_size
        self.gradient_clipping = gradient_clipping
        self.pre_layer_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="pre_layer_norm",
            dtype=self.dtype_policy,
        )
        self.linear_start = keras.layers.Dense(
            hidden_size * 2,
            use_bias=False,
            name="linear_start",
            dtype=self.dtype_policy,
        )
        self.depthwise_conv1d = keras.layers.DepthwiseConv1D(
            kernel_size=conf_conv_kernel_size,
            strides=1,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            name="depthwise_conv1d",
            dtype=self.dtype_policy,
        )
        self.conv_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="conv_norm",
            dtype=self.dtype_policy,
        )
        self.linear_end = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="linear_end",
            dtype=self.dtype_policy,
        )
        self.causal_padding = conf_conv_kernel_size - 1

    def build(self, input_shape):
        self.pre_layer_norm.build(input_shape)
        self.linear_start.build(input_shape)
        glu_output_shape = input_shape[:-1] + (self.hidden_size,)
        self.depthwise_conv1d.build(glu_output_shape)
        self.conv_norm.build(glu_output_shape)
        self.linear_end.build(glu_output_shape)
        super().build(input_shape)

    def call(self, audio_encodings):
        residual = audio_encodings
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)
        gated, activated = keras.ops.split(audio_encodings, 2, axis=-1)
        audio_encodings = gated * keras.activations.sigmoid(activated)

        padded = keras.ops.pad(
            audio_encodings,
            [[0, 0], [self.causal_padding, 0], [0, 0]],
        )
        audio_encodings = self.depthwise_conv1d(padded)
        audio_encodings = keras.ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = keras.activations.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        return audio_encodings + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "conf_conv_kernel_size": self.conf_conv_kernel_size,
                "gradient_clipping": self.gradient_clipping,
            }
        )
        return config


class Gemma3nAudioConformerAttention(keras.layers.Layer):
    """The attention module for the Conformer block.

    This module implements the multi-head self-attention sub-layer of a
    Conformer block. It wraps the core attention mechanism with pre and post
    layer normalization, a final dense projection, and a residual connection.

    Args:
        hidden_size: int. The hidden size of the input and output tensors.
        gradient_clipping: float. The maximum absolute value for gradient
            clipping.
        conf_num_attention_heads: int. The number of attention heads.
        conf_attention_chunk_size: int. The chunk size for attention
            computation, used for memory efficiency.
        conf_attention_context_right: int. The right context size for attention.
        conf_attention_context_left: int. The left context size for attention.
        conf_attention_logit_cap: float. The value to which attention logits
            are capped.
    """

    def __init__(
        self,
        hidden_size,
        gradient_clipping,
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
        self.gradient_clipping = gradient_clipping
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.pre_attn_norm = Gemma3nRMSNorm(
            hidden_size, name="pre_attn_norm", dtype=self.dtype_policy
        )
        self.attn = Gemma3nAudioAttention(
            hidden_size,
            conf_num_attention_heads,
            conf_attention_chunk_size,
            conf_attention_context_right,
            conf_attention_context_left,
            conf_attention_logit_cap,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.post = keras.layers.Dense(
            hidden_size, use_bias=False, name="post", dtype=self.dtype_policy
        )
        self.post_norm = Gemma3nRMSNorm(
            hidden_size, name="post_norm", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        self.pre_attn_norm.build(input_shape)
        self.attn.build(input_shape)
        self.post.build(input_shape)
        self.post_norm.build(input_shape)
        super().build(input_shape)

    def call(self, audio_encodings, audio_mel_mask):
        residual = audio_encodings
        audio_encodings = keras.ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        audio_encodings_attn_out = self.attn(
            audio_encodings_norm, audio_mel_mask
        )
        b, t, num_heads, head_dim = keras.ops.shape(audio_encodings_attn_out)
        audio_encodings_reshaped = keras.ops.reshape(
            audio_encodings_attn_out, (b, t, num_heads * head_dim)
        )
        audio_encodings = self.post(audio_encodings_reshaped)
        audio_encodings = keras.ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return residual + self.post_norm(audio_encodings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "gradient_clipping": self.gradient_clipping,
                "conf_num_attention_heads": self.conf_num_attention_heads,
                "conf_attention_chunk_size": self.conf_attention_chunk_size,
                "conf_attention_context_right": self.conf_attention_context_right,  # noqa: E501
                "conf_attention_context_left": self.conf_attention_context_left,
                "conf_attention_logit_cap": self.conf_attention_logit_cap,
            }
        )
        return config
