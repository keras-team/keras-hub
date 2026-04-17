import keras
from keras import ops

from keras_hub.src.models.gemma3n.gemma3n_audio_layers import (
    Gemma3nAudioConformerAttention,
)
from keras_hub.src.models.gemma3n.gemma3n_audio_layers import (
    Gemma3nAudioConformerFeedForward,
)
from keras_hub.src.models.gemma3n.gemma3n_audio_layers import (
    Gemma3nAudioConformerLightConv1d,
)
from keras_hub.src.models.gemma3n.gemma3n_audio_layers import (
    Gemma3nAudioSSCPConvBlock,
)
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nAudioSubSampleConvProjection(keras.layers.Layer):
    """A convolutional projection layer that subsamples audio features.

    This layer applies two blocks of 2D convolutions to the input audio
    spectrogram. Each block subsamples the input along the time and frequency
    dimensions. The output is then flattened and projected to the model's
    hidden size.

    Args:
        input_feat_size: int. The number of frequency bins in the input
            spectrogram.
        hidden_size: int. The dimensionality of the output embeddings.
        sscp_conv_channel_size: list of int. The number of output channels for
            each of the two convolutional blocks.
        sscp_conv_kernel_size: list of tuple of int. The kernel sizes for each
            of the two convolutional blocks.
        sscp_conv_stride_size: list of tuple of int. The stride sizes for each
            of the two convolutional blocks.
        sscp_conv_group_norm_eps: float. Epsilon value for the Group
            Normalization layers within the convolutional blocks.
    """

    def __init__(
        self,
        input_feat_size,
        hidden_size,
        sscp_conv_channel_size,
        sscp_conv_kernel_size,
        sscp_conv_stride_size,
        sscp_conv_group_norm_eps,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.input_feat_size = input_feat_size
        self.sscp_conv_channel_size = sscp_conv_channel_size
        self.sscp_conv_kernel_size = sscp_conv_kernel_size
        self.sscp_conv_stride_size = sscp_conv_stride_size
        self.sscp_conv_group_norm_eps = sscp_conv_group_norm_eps

        # as per model specifics, we always perform 2 convolutions
        if len(sscp_conv_channel_size) != 2:
            raise ValueError(
                f"sscp_conv_channel_size must have exactly 2 elements, "
                f"got {len(sscp_conv_channel_size)}"
            )
        if len(sscp_conv_kernel_size) != 2:
            raise ValueError(
                f"sscp_conv_kernel_size must have exactly 2 elements, "
                f"got {len(sscp_conv_kernel_size)}"
            )
        if len(sscp_conv_stride_size) != 2:
            raise ValueError(
                f"sscp_conv_stride_size must have exactly 2 elements, "
                f"got {len(sscp_conv_stride_size)}"
            )

        current_f_for_block_input = input_feat_size
        self.calculated_block_padding = []
        self.calculated_f_out_dims = []
        for i in range(2):
            kernel_h, kernel_w = sscp_conv_kernel_size[i]
            _, stride_w = sscp_conv_stride_size[i]
            pad_t_top, pad_t_bottom, pad_f_left, pad_f_right = (
                0,
                kernel_h - 1,
                1,
                1,
            )
            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            self.calculated_block_padding.append(manual_padding_tuple)
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1
            self.calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv
        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            input_freq_dim=input_feat_size,
            out_channels=sscp_conv_channel_size[0],
            kernel_size=sscp_conv_kernel_size[0],
            stride_size=sscp_conv_stride_size[0],
            group_norm_eps=sscp_conv_group_norm_eps,
            manual_padding=self.calculated_block_padding[0],
            name="conv_0",
            dtype=self.dtype_policy,
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            input_freq_dim=self.calculated_f_out_dims[0],
            out_channels=sscp_conv_channel_size[1],
            kernel_size=sscp_conv_kernel_size[1],
            stride_size=sscp_conv_stride_size[1],
            group_norm_eps=sscp_conv_group_norm_eps,
            manual_padding=self.calculated_block_padding[1],
            name="conv_1",
            dtype=self.dtype_policy,
        )
        self.input_proj_linear = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="input_proj_linear",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        _, t_in, f_in = input_shape
        conv0_input_shape = (None, 1, t_in, f_in)
        self.conv_0.build(conv0_input_shape)
        if t_in is not None:
            pad_t_top_0, pad_t_bottom_0 = self.calculated_block_padding[0][2:4]
            kernel_h_0, _ = self.sscp_conv_kernel_size[0]
            stride_h_0, _ = self.sscp_conv_stride_size[0]
            t_padded_0 = t_in + pad_t_top_0 + pad_t_bottom_0
            t_out_0 = (t_padded_0 - kernel_h_0) // stride_h_0 + 1
        else:
            t_out_0 = None
        c_out_0 = self.sscp_conv_channel_size[0]
        f_out_0 = self.calculated_f_out_dims[0]
        conv1_input_shape = (None, c_out_0, t_out_0, f_out_0)
        self.conv_1.build(conv1_input_shape)
        if t_out_0 is not None:
            t_padded_1 = (
                t_out_0
                + self.calculated_block_padding[1][2]
                + self.calculated_block_padding[1][3]
            )
            kernel_h_1, _ = self.sscp_conv_kernel_size[1]
            stride_h_1, _ = self.sscp_conv_stride_size[1]
            t_out_1 = (t_padded_1 - kernel_h_1) // stride_h_1 + 1
        else:
            t_out_1 = None
        c_out_1 = self.sscp_conv_channel_size[1]
        f_out_1 = self.calculated_f_out_dims[1]
        proj_input_shape = (None, t_out_1, f_out_1 * c_out_1)
        self.input_proj_linear.build(proj_input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        b, t_in, f_in = input_shape
        if t_in is not None:
            _, _, pad_t_top_0, pad_t_bottom_0 = self.calculated_block_padding[0]
            kernel_h_0, _ = self.sscp_conv_kernel_size[0]
            stride_h_0, _ = self.sscp_conv_stride_size[0]
            t_padded_0 = t_in + pad_t_top_0 + pad_t_bottom_0
            t_out_0 = (t_padded_0 - kernel_h_0) // stride_h_0 + 1
            _, _, pad_t_top_1, pad_t_bottom_1 = self.calculated_block_padding[1]
            kernel_h_1, _ = self.sscp_conv_kernel_size[1]
            stride_h_1, _ = self.sscp_conv_stride_size[1]
            t_padded_1 = t_out_0 + pad_t_top_1 + pad_t_bottom_1
            t_out_1 = (t_padded_1 - kernel_h_1) // stride_h_1 + 1
        else:
            t_out_1 = None
        return (b, t_out_1, self.hidden_size)

    def call(self, audio_encodings):
        audio_encodings_reshaped = ops.expand_dims(audio_encodings, 1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        b, c_out, t_out, f_out = ops.shape(x)
        x_permuted = ops.transpose(x, (0, 2, 3, 1))
        output_flattened = ops.reshape(x_permuted, (b, t_out, f_out * c_out))
        return self.input_proj_linear(output_flattened)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_feat_size": self.input_feat_size,
                "hidden_size": self.hidden_size,
                "sscp_conv_channel_size": self.sscp_conv_channel_size,
                "sscp_conv_kernel_size": self.sscp_conv_kernel_size,
                "sscp_conv_stride_size": self.sscp_conv_stride_size,
                "sscp_conv_group_norm_eps": self.sscp_conv_group_norm_eps,
            }
        )
        return config


class Gemma3nAudioConformerBlock(keras.layers.Layer):
    """A single conformer block for processing audio sequences.

    This layer implements the conformer architecture, which consists of a
    sequence of four modules: a feed-forward module, a multi-head
    self-attention module, a convolution module, and a final feed-forward
    module. The output of each module is added to its input through a residual
    connection.

    Args:
        hidden_size: int. The dimensionality of the input and output embeddings.
        rms_norm_eps: float. Epsilon value for the Gemma 3n RMS normalization
            layers.
        gradient_clipping: float. The maximum absolute value for the gradient.
        residual_weight: float. The weight for the residual connection in
            the feed-forward layers.
        num_attention_heads: int. The number of attention heads.
        attention_chunk_size: int. The size of chunks for local attention.
        num_attention_context_right: int. The right context size for local
            attention.
        num_attention_context_left: int. The left context size for local
            attention.
        attention_logit_cap: float. The maximum value for the attention
            logits.
        conv_kernel_size: int. The kernel size for the 1D convolution
            layer.
    """

    def __init__(
        self,
        hidden_size,
        rms_norm_eps,
        gradient_clipping,
        residual_weight,
        num_attention_heads,
        attention_chunk_size,
        num_attention_context_right,
        num_attention_context_left,
        attention_logit_cap,
        conv_kernel_size,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.gradient_clipping = gradient_clipping
        self.residual_weight = residual_weight
        self.num_attention_heads = num_attention_heads
        self.attention_chunk_size = attention_chunk_size
        self.num_attention_context_right = num_attention_context_right
        self.num_attention_context_left = num_attention_context_left
        self.attention_logit_cap = attention_logit_cap
        self.conv_kernel_size = conv_kernel_size
        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(
            hidden_size=hidden_size,
            gradient_clipping=gradient_clipping,
            residual_weight=residual_weight,
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype_policy,
            name="ffw_layer_start",
        )
        self.attention = Gemma3nAudioConformerAttention(
            hidden_size=hidden_size,
            gradient_clipping=gradient_clipping,
            num_attention_heads=num_attention_heads,
            attention_chunk_size=attention_chunk_size,
            num_attention_context_right=num_attention_context_right,
            num_attention_context_left=num_attention_context_left,
            attention_logit_cap=attention_logit_cap,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(
            hidden_size=hidden_size,
            rms_norm_eps=rms_norm_eps,
            conv_kernel_size=conv_kernel_size,
            gradient_clipping=gradient_clipping,
            dtype=self.dtype_policy,
            name="lconv1d",
        )
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(
            hidden_size=hidden_size,
            gradient_clipping=gradient_clipping,
            residual_weight=residual_weight,
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype_policy,
            name="ffw_layer_end",
        )
        self.norm = Gemma3nRMSNorm(
            hidden_size, eps=rms_norm_eps, name="norm", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        if (
            isinstance(input_shape, tuple)
            and len(input_shape) == 2
            and isinstance(input_shape[0], tuple)
        ):
            audio_encodings_shape, _ = input_shape
        elif isinstance(input_shape, tuple) and len(input_shape) >= 3:
            audio_encodings_shape = input_shape
        else:
            raise ValueError(
                f"Unexpected `input_shape` structure for "
                f"Gemma3nAudioConformerBlock: {input_shape}"
            )
        self.ffw_layer_start.build(audio_encodings_shape)
        self.attention.build(audio_encodings_shape)
        self.lconv1d.build(audio_encodings_shape)
        self.ffw_layer_end.build(audio_encodings_shape)
        self.norm.build(audio_encodings_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if (
            isinstance(input_shape, tuple)
            and len(input_shape) == 2
            and isinstance(input_shape[0], tuple)
        ):
            audio_encodings_shape, _ = input_shape
            return audio_encodings_shape
        elif isinstance(input_shape, tuple) and len(input_shape) >= 3:
            return input_shape
        else:
            raise ValueError(
                f"Unexpected `input_shape` structure for "
                f"Gemma3nAudioConformerBlock: {input_shape}"
            )

    def call(self, inputs):
        audio_encodings, audio_mel_mask = inputs
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ops.logical_not(audio_mel_mask)
        mask_shape = ops.shape(validity_mask_for_lconv)
        enc_shape = ops.shape(audio_encodings)
        if len(mask_shape) < len(enc_shape):
            validity_mask_for_lconv = ops.expand_dims(
                validity_mask_for_lconv, -1
            )
        audio_encodings_for_lconv_input = audio_encodings * ops.cast(
            validity_mask_for_lconv,
            audio_encodings.dtype,
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)
        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = ops.clip(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        output = self.norm(audio_encodings)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "gradient_clipping": self.gradient_clipping,
                "residual_weight": self.residual_weight,
                "num_attention_heads": self.num_attention_heads,
                "attention_chunk_size": self.attention_chunk_size,
                "num_attention_context_right": self.num_attention_context_right,  # noqa: E501
                "num_attention_context_left": self.num_attention_context_left,
                "attention_logit_cap": self.attention_logit_cap,
                "conv_kernel_size": self.conv_kernel_size,
            }
        )
        return config


class Gemma3nAudioEncoder(keras.layers.Layer):
    """The main audio encoder for the Gemma3n model.

    This layer combines a subsampling convolutional projection with a stack of
    conformer blocks to encode audio spectrograms into a sequence of hidden
    states.

    Args:
        hidden_size: int. The dimensionality of the embeddings.
        input_feat_size: int. The number of frequency bins in the input
            spectrogram.
        sscp_conv_channel_size: list of int. The number of output channels for
            each of the two convolutional blocks in the subsampler.
        sscp_conv_kernel_size: list of tuple of int. The kernel sizes for each
            of the two convolutional blocks in the subsampler.
        sscp_conv_stride_size: list of tuple of int. The stride sizes for each
            of the two convolutional blocks in the subsampler.
        sscp_conv_group_norm_eps: float. Epsilon value for the Group
            Normalization layers in the subsampler.
        num_hidden_layers: int. The number of conformer blocks.
        rms_norm_eps: float. Epsilon value for the Gemma 3n RMS normalization
            layers.
        gradient_clipping: float. The maximum absolute value for the gradient.
        residual_weight: float. The weight for the residual connection in
            the feed-forward layers of the conformer blocks.
        num_attention_heads: int. The number of attention heads in the
            conformer blocks.
        attention_chunk_size: int. The size of chunks for local attention
            in the conformer blocks.
        num_attention_context_right: int. The right context size for local
            attention in the conformer blocks.
        num_attention_context_left: int. The left context size for local
            attention in the conformer blocks.
        attention_logit_cap: float. The maximum value for the attention
            logits in the conformer blocks.
        conv_kernel_size: int. The kernel size for the 1D convolution
            layer in the conformer blocks.
        reduction_factor: int. The factor by which to reduce the sequence
            length of the final output.
    """

    def __init__(
        self,
        hidden_size,
        input_feat_size,
        sscp_conv_channel_size,
        sscp_conv_kernel_size,
        sscp_conv_stride_size,
        sscp_conv_group_norm_eps,
        num_hidden_layers,
        rms_norm_eps,
        gradient_clipping,
        residual_weight,
        num_attention_heads,
        attention_chunk_size,
        num_attention_context_right,
        num_attention_context_left,
        attention_logit_cap,
        conv_kernel_size,
        reduction_factor,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.input_feat_size = input_feat_size
        self.sscp_conv_channel_size = sscp_conv_channel_size
        self.sscp_conv_kernel_size = sscp_conv_kernel_size
        self.sscp_conv_stride_size = sscp_conv_stride_size
        self.sscp_conv_group_norm_eps = sscp_conv_group_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.gradient_clipping = gradient_clipping
        self.residual_weight = residual_weight
        self.num_attention_heads = num_attention_heads
        self.attention_chunk_size = attention_chunk_size
        self.num_attention_context_right = num_attention_context_right
        self.num_attention_context_left = num_attention_context_left
        self.attention_logit_cap = attention_logit_cap
        self.conv_kernel_size = conv_kernel_size
        self.reduction_factor = reduction_factor
        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(
            input_feat_size,
            hidden_size,
            sscp_conv_channel_size,
            sscp_conv_kernel_size,
            sscp_conv_stride_size,
            sscp_conv_group_norm_eps,
            dtype=self.dtype_policy,
            name="subsample_conv_projection",
        )
        self.conformer = [
            Gemma3nAudioConformerBlock(
                hidden_size,
                rms_norm_eps,
                gradient_clipping,
                residual_weight,
                num_attention_heads,
                attention_chunk_size,
                num_attention_context_right,
                num_attention_context_left,
                attention_logit_cap,
                conv_kernel_size,
                dtype=self.dtype_policy,
                name=f"conformer_block_{i}",
            )
            for i in range(num_hidden_layers)
        ]

    def build(self, input_shape):
        if (
            isinstance(input_shape, tuple)
            and len(input_shape) == 2
            and isinstance(input_shape[0], tuple)
        ):
            audio_mel_shape, _ = input_shape
        else:
            raise ValueError(
                f"Unexpected `input_shape` structure for Gemma3nAudioEncoder: "
                f"{input_shape}"
            )
        self.subsample_conv_projection.build(audio_mel_shape)
        encodings_shape = self.subsample_conv_projection.compute_output_shape(
            audio_mel_shape
        )
        t_sub = encodings_shape[1]
        batch_size = audio_mel_shape[0]
        current_mask_shape = (
            (batch_size, t_sub) if t_sub is not None else (batch_size, None)
        )
        current_encodings_shape = encodings_shape
        for block in self.conformer:
            block.build((current_encodings_shape, current_mask_shape))
            current_encodings_shape = block.compute_output_shape(
                (current_encodings_shape, current_mask_shape)
            )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        audio_mel_shape, _ = input_shape
        encodings_shape = self.subsample_conv_projection.compute_output_shape(
            audio_mel_shape
        )
        t_sub = encodings_shape[1]
        batch_size = audio_mel_shape[0]
        current_mask_shape = (
            (batch_size, t_sub) if t_sub is not None else (batch_size, None)
        )
        current_encodings_shape = encodings_shape
        for block in self.conformer:
            current_encodings_shape = block.compute_output_shape(
                (current_encodings_shape, current_mask_shape)
            )
        final_mask_shape = current_mask_shape
        if self.reduction_factor > 1:
            t_sub = current_encodings_shape[1]
            if t_sub is not None:
                new_t = t_sub // self.reduction_factor
                current_encodings_shape = (
                    current_encodings_shape[0],
                    new_t,
                    current_encodings_shape[2],
                )
                final_mask_shape = (
                    (current_mask_shape[0], new_t)
                    if current_mask_shape[1] is not None
                    else (current_mask_shape[0], None)
                )
        return current_encodings_shape, final_mask_shape

    def call(self, inputs):
        audio_mel, audio_mel_mask = inputs
        audio_encodings = self.subsample_conv_projection(audio_mel)
        t_sub = ops.shape(audio_encodings)[1]
        time_stride_product = 1
        for stride_pair in self.sscp_conv_stride_size:
            time_stride_product *= stride_pair[0]
        mask_rank = len(ops.shape(audio_mel_mask))
        audio_mel_mask_to_take = audio_mel_mask
        if mask_rank > 2:
            audio_mel_mask_to_take = ops.squeeze(
                audio_mel_mask, axis=list(range(1, mask_rank - 1))
            )
        indices = ops.arange(0, t_sub) * time_stride_product
        indices = ops.clip(indices, 0, ops.shape(audio_mel_mask_to_take)[1] - 1)
        current_mask = ops.take(audio_mel_mask_to_take, indices, axis=1)
        for block in self.conformer:
            audio_encodings = block((audio_encodings, current_mask))

        if self.reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.reduction_factor]
            current_mask = current_mask[:, :: self.reduction_factor]
        mask_shape = ops.shape(current_mask)
        enc_shape = ops.shape(audio_encodings)
        if len(mask_shape) < len(enc_shape):
            current_mask_expanded = ops.expand_dims(current_mask, axis=-1)
        else:
            current_mask_expanded = current_mask
        return audio_encodings * ops.cast(
            ops.logical_not(current_mask_expanded),
            audio_encodings.dtype,
        ), current_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "input_feat_size": self.input_feat_size,
                "sscp_conv_channel_size": self.sscp_conv_channel_size,
                "sscp_conv_kernel_size": self.sscp_conv_kernel_size,
                "sscp_conv_stride_size": self.sscp_conv_stride_size,
                "sscp_conv_group_norm_eps": self.sscp_conv_group_norm_eps,
                "num_hidden_layers": self.num_hidden_layers,
                "rms_norm_eps": self.rms_norm_eps,
                "gradient_clipping": self.gradient_clipping,
                "residual_weight": self.residual_weight,
                "num_attention_heads": self.num_attention_heads,
                "attention_chunk_size": self.attention_chunk_size,
                "num_attention_context_right": self.num_attention_context_right,  # noqa: E501
                "num_attention_context_left": self.num_attention_context_left,
                "attention_logit_cap": self.attention_logit_cap,
                "conv_kernel_size": self.conv_kernel_size,
                "reduction_factor": self.reduction_factor,
            }
        )
        return config
