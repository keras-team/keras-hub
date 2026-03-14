import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen3ASREncoderLayer(keras.layers.Layer):
    """Transformer encoder layer for the Qwen3-ASR audio encoder.

    Implements a pre-norm transformer block with multi-head self-attention
    and a GELU feedforward network.

    Args:
        d_model: int. Hidden size of the encoder.
        num_heads: int. Number of attention heads.
        ffn_dim: int. Intermediate dimension of the feedforward network.
        dropout: float. Dropout rate for attention weights. Defaults to 0.0.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.head_dim = d_model // num_heads

    def build(self, input_shape):
        # Self-attention sublayers.
        self.self_attn_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="self_attn_layer_norm",
        )
        self.self_attn_layer_norm.build(input_shape)

        self.q_proj = keras.layers.Dense(
            self.d_model,
            use_bias=True,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.q_proj.build(input_shape)

        self.k_proj = keras.layers.Dense(
            self.d_model,
            use_bias=True,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.k_proj.build(input_shape)

        self.v_proj = keras.layers.Dense(
            self.d_model,
            use_bias=True,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.v_proj.build(input_shape)

        attn_out_shape = list(input_shape)
        attn_out_shape[-1] = self.d_model
        self.out_proj = keras.layers.Dense(
            self.d_model,
            use_bias=True,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(tuple(attn_out_shape))

        self.attn_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        # Feedforward sublayers.
        self.final_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="final_layer_norm",
        )
        self.final_layer_norm.build(input_shape)

        self.fc1 = keras.layers.Dense(
            self.ffn_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc1.build(input_shape)

        fc2_shape = list(input_shape)
        fc2_shape[-1] = self.ffn_dim
        self.fc2 = keras.layers.Dense(
            self.d_model,
            use_bias=True,
            dtype=self.dtype_policy,
            name="fc2",
        )
        self.fc2.build(tuple(fc2_shape))

        self.built = True

    def call(self, hidden_states, attention_mask=None, training=None):
        # Self-attention with pre-norm and residual.
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, seq, heads, head_dim).
        query = ops.reshape(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        key = ops.reshape(
            key, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        value = ops.reshape(
            value, (batch_size, seq_len, self.num_heads, self.head_dim)
        )

        # Transpose to (batch, heads, seq, head_dim).
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (
            ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(ops.cast(attn_weights, "float32"), axis=-1)
        attn_weights = ops.cast(attn_weights, self.compute_dtype)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, value)

        # Transpose back and reshape to (batch, seq, d_model).
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.d_model)
        )

        hidden_states = self.out_proj(attn_output)
        hidden_states = residual + hidden_states

        # Feedforward with pre-norm and residual.
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = keras.activations.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen3ASREncoder(keras.layers.Layer):
    """Audio Transformer (AuT) encoder for Qwen3-ASR.

    Processes mel spectrogram features through Conv2D downsampling (8x) and
    transformer encoder layers, producing hidden states for the Qwen3 text
    decoder.

    The encoder consists of:
    1. Three Conv2D layers with stride 2 and GELU activation, providing 8x
       downsampling in both time and frequency dimensions.
    2. Linear projection from the flattened conv output to the model dimension.
    3. Scaling by ``sqrt(d_model)``.
    4. A stack of transformer encoder layers, each with standard multi-head
       self-attention and a GELU feedforward network.
    5. Post-encoder layer normalization.
    6. Optional two-layer projection (Dense, GELU, Dense) to map encoder
       outputs to the text decoder dimension.

    Args:
        num_mel_bins: int. Number of mel frequency bins in the input
            spectrogram. Defaults to 128.
        d_model: int. Hidden size of the transformer encoder.
            Defaults to 1024.
        encoder_layers: int. Number of transformer encoder layers.
            Defaults to 24.
        encoder_attention_heads: int. Number of attention heads.
            Defaults to 16.
        encoder_ffn_dim: int. Intermediate dimension of the feedforward
            network in each encoder layer. Defaults to 4096.
        downsample_hidden_size: int. Number of filters for the Conv2D
            downsampling layers. Defaults to 480.
        output_dim: int or None. If set, adds a projection to map encoder
            outputs to this dimension. Typically the text decoder hidden
            size. Defaults to None.
        dropout: float. Dropout rate for attention weights.
            Defaults to 0.0.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_mel_bins=128,
        d_model=1024,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        downsample_hidden_size=480,
        output_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.num_encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.output_dim = output_dim
        self.dropout = dropout
        self.embed_scale = math.sqrt(d_model)

        # Compute frequency dimension after three stride-2 convolutions.
        freq = num_mel_bins
        for _ in range(3):
            freq = (freq + 1) // 2
        self._freq_after_conv = freq

    def build(self, input_shape):
        # Conv2D downsampling layers.
        self.conv2d_1 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=self.dtype_policy,
            name="conv2d_1",
        )
        self.conv2d_2 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=self.dtype_policy,
            name="conv2d_2",
        )
        self.conv2d_3 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=self.dtype_policy,
            name="conv2d_3",
        )

        # Linear projection from flattened conv output to d_model.
        conv_flat_dim = self.downsample_hidden_size * self._freq_after_conv
        self.conv_projection = keras.layers.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype_policy,
            name="conv_projection",
        )

        # Transformer encoder layers.
        self._encoder_layers = []
        for i in range(self.num_encoder_layers):
            layer = Qwen3ASREncoderLayer(
                d_model=self.d_model,
                num_heads=self.encoder_attention_heads,
                ffn_dim=self.encoder_ffn_dim,
                dropout=self.dropout,
                dtype=self.dtype_policy,
                name=f"encoder_layer_{i}",
            )
            self._encoder_layers.append(layer)

        # Post-encoder layer normalization.
        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="layer_norm",
        )

        # Optional output projection.
        if self.output_dim is not None:
            self.output_proj_1 = keras.layers.Dense(
                self.d_model,
                use_bias=True,
                dtype=self.dtype_policy,
                name="output_proj_1",
            )
            self.output_proj_2 = keras.layers.Dense(
                self.output_dim,
                use_bias=True,
                dtype=self.dtype_policy,
                name="output_proj_2",
            )

        # Build sublayers. Input shape: (batch, time, mel_bins).
        conv_shape = (input_shape[0], input_shape[1], self.num_mel_bins, 1)
        self.conv2d_1.build(conv_shape)
        conv1_out = self.conv2d_1.compute_output_shape(conv_shape)
        self.conv2d_2.build(conv1_out)
        conv2_out = self.conv2d_2.compute_output_shape(conv1_out)
        self.conv2d_3.build(conv2_out)

        flat_shape = (input_shape[0], None, conv_flat_dim)
        self.conv_projection.build(flat_shape)

        transformer_shape = (input_shape[0], None, self.d_model)
        for layer in self._encoder_layers:
            layer.build(transformer_shape)

        self.layer_norm.build(transformer_shape)

        if self.output_dim is not None:
            self.output_proj_1.build(transformer_shape)
            self.output_proj_2.build(transformer_shape)

        self.built = True

    def call(self, input_features, attention_mask=None, training=None):
        """Forward pass for the Qwen3-ASR audio encoder.

        Args:
            input_features: Float tensor of shape
                ``(batch_size, time_steps, num_mel_bins)`` containing mel
                spectrogram features.
            attention_mask: Optional float tensor broadcastable to
                ``(batch_size, 1, seq_len, seq_len)`` applied after Conv2D
                downsampling, where 0 indicates positions to attend and
                large negative values indicate positions to mask.
            training: Boolean indicating training mode.

        Returns:
            Encoder hidden states of shape
            ``(batch_size, time_steps // 8, output_dim or d_model)``.
        """
        # Add channel dimension: (batch, time, mel) -> (batch, time, mel, 1).
        x = ops.expand_dims(input_features, axis=-1)

        # Conv2D downsampling with GELU activation.
        x = keras.activations.gelu(self.conv2d_1(x))
        x = keras.activations.gelu(self.conv2d_2(x))
        x = keras.activations.gelu(self.conv2d_3(x))

        # Flatten frequency and channel dimensions.
        # (batch, T//8, F//8, C) -> (batch, T//8, F//8 * C)
        batch_size = ops.shape(x)[0]
        time_steps = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, time_steps, -1))

        # Project to d_model and scale.
        hidden_states = self.conv_projection(x)
        hidden_states = hidden_states * ops.cast(
            self.embed_scale, hidden_states.dtype
        )

        # Transformer encoder layers.
        for layer in self._encoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
            )

        # Post-encoder layer normalization.
        hidden_states = self.layer_norm(hidden_states)

        # Optional output projection.
        if self.output_dim is not None:
            hidden_states = keras.activations.gelu(
                self.output_proj_1(hidden_states)
            )
            hidden_states = self.output_proj_2(hidden_states)

        return hidden_states

    def compute_output_shape(self, input_shape):
        time_steps = input_shape[1]
        if time_steps is not None:
            for _ in range(3):
                time_steps = (time_steps + 1) // 2
        out_dim = (
            self.output_dim if self.output_dim is not None else self.d_model
        )
        return (input_shape[0], time_steps, out_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mel_bins": self.num_mel_bins,
                "d_model": self.d_model,
                "encoder_layers": self.num_encoder_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "downsample_hidden_size": self.downsample_hidden_size,
                "output_dim": self.output_dim,
                "dropout": self.dropout,
            }
        )
        return config
