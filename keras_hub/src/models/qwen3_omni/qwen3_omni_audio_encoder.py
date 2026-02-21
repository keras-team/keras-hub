import math

import keras
import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export


def _create_sinusoidal_positions(length, channels, max_timescale=10000):
    """Create fixed sinusoidal positional embeddings."""

    if channels % 2 != 0:
        raise ValueError(
            "Sinusoidal position embeddings require even channels. "
            f"Received channels={channels}"
        )
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(
        -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
    )
    scaled_time = (
        np.arange(length, dtype=np.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )

    positional_embedding = np.concatenate(
        [np.sin(scaled_time), np.cos(scaled_time)], axis=1
    )
    return positional_embedding


@keras_hub_export("keras_hub.models.Qwen3OmniAudioEncoder")
class Qwen3OmniAudioEncoder(keras.layers.Layer):
    """Audio encoder for Qwen3-Omni

    This encoder processes mel-spectrogram audio features. It includes:

    - Convolutional downsampling (3 Conv2D layers, 8x total reduction)
    - Fixed sinusoidal positional embeddings
    - Transformer encoder layers
    - Output projection to match text model dimension

    Args:
        num_mel_bins: int. The number of mel frequency bins. Defaults to `128`.
        d_model: int. The model dimension (hidden size). Defaults to `1280`.
        encoder_layers: int. The number of transformer encoder layers.
            Defaults to `32`.
        encoder_attention_heads: int. The number of attention heads.
            Defaults to `20`.
        encoder_ffn_dim: int. The feed-forward network dimension.
            Defaults to `5120`.
        output_dim: int. The output projection dimension (should match text
            model hidden dimension). Defaults to `2048`.
        downsample_hidden_size: int. The hidden size for convolutional
            downsampling layers. Defaults to `480`.
        max_source_positions: int. The maximum sequence length after
            downsampling. Defaults to `1500`.
        scale_embedding: bool. Whether to scale embeddings by sqrt(d_model).
            Defaults to `False`.
        activation_function: string. The activation function name.
            Defaults to `"gelu"`.
        dropout: float. The dropout rate. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Mel-spectrogram input (batch_size, time_steps, mel_bins)
    input_features = np.random.uniform(size=(1, 3000, 128))

    # Audio encoder
    audio_encoder = keras_hub.models.Qwen3OmniAudioEncoder(
        num_mel_bins=128,
        d_model=1280,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        output_dim=2048,
    )
    output = audio_encoder({"input_features": input_features})
    ```
    """

    def __init__(
        self,
        num_mel_bins=128,
        d_model=1280,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        output_dim=2048,
        downsample_hidden_size=480,
        max_source_positions=1500,
        scale_embedding=False,
        activation_function="gelu",
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers_count = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.output_dim = output_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.max_source_positions = max_source_positions
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.dropout_rate = dropout

        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0

        # === Convolutional downsampling layers ===
        self.conv2d1 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=dtype,
            name="conv2d1",
        )
        self.conv2d2 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=dtype,
            name="conv2d2",
        )
        self.conv2d3 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="same",
            dtype=dtype,
            name="conv2d3",
        )

        self.conv_out = layers.Dense(
            d_model,
            use_bias=False,
            dtype=dtype,
            name="conv_out",
        )

        self._positional_embedding_np = _create_sinusoidal_positions(
            max_source_positions, d_model
        )

        # === Transformer encoder layers ===
        self.encoder_layers = [
            Qwen3OmniAudioEncoderLayer(
                embed_dim=d_model,
                num_heads=encoder_attention_heads,
                ffn_dim=encoder_ffn_dim,
                activation=activation_function,
                dropout=dropout,
                dtype=dtype,
                name=f"encoder_layer_{i}",
            )
            for i in range(encoder_layers)
        ]

        # === Post-encoder normalization ===
        self.ln_post = layers.LayerNormalization(
            dtype=dtype,
            name="layer_norm",
        )

        # === Output Projection ===
        self.proj1 = layers.Dense(
            d_model,
            use_bias=True,
            dtype=dtype,
            name="proj1",
        )
        self.proj_activation = layers.Activation(
            activation_function, dtype=dtype
        )
        self.proj2 = layers.Dense(
            output_dim,
            use_bias=True,
            dtype=dtype,
            name="proj2",
        )
        self.dropout_layer = layers.Dropout(
            dropout, dtype=dtype, name="dropout"
        )

    def _call_with_inputs(self, input_features, training=False):
        """Encode mel-spectrogram features into output embeddings.

        Args:
            input_features: Tensor with shape
                `(batch_size, time_steps, num_mel_bins)`.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(batch_size, seq_len, output_dim)`.
        """
        # Apply convolutional downsampling
        # Input: (batch, time, mel_bins) -> (batch, time, mel_bins, 1)
        hidden_states = ops.expand_dims(input_features, axis=-1)

        hidden_states = self.conv2d1(hidden_states, training=training)
        hidden_states = ops.gelu(hidden_states)

        hidden_states = self.conv2d2(hidden_states, training=training)
        hidden_states = ops.gelu(hidden_states)

        hidden_states = self.conv2d3(hidden_states, training=training)
        hidden_states = ops.gelu(hidden_states)

        # Flatten spatial dimensions and project
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]
        hidden_states = ops.transpose(hidden_states, [0, 1, 3, 2])
        hidden_states = ops.reshape(hidden_states, [batch_size, seq_len, -1])
        hidden_states = self.conv_out(hidden_states)

        # Scale embeddings
        hidden_states = hidden_states * self.embed_scale

        # Add position embeddings
        pos_embed_tensor = ops.convert_to_tensor(
            self._positional_embedding_np, dtype=self.compute_dtype
        )
        positions = pos_embed_tensor[:seq_len, :]
        hidden_states = hidden_states + positions

        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(
                hidden_states,
                training=training,
            )

        # Post-encoder normalization
        hidden_states = self.ln_post(hidden_states)

        # Output projection
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.proj_activation(hidden_states)
        hidden_states = self.proj2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)

        return hidden_states

    def call(self, inputs, training=False):
        """Process a dict of inputs through the audio encoder.

        Args:
            inputs: dict. A dictionary with `"input_features"` key containing
                mel-spectrogram tensor with shape
                `(batch_size, time_steps, num_mel_bins)`.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(batch_size, seq_len, output_dim)`.
        """
        return self._call_with_inputs(
            inputs["input_features"], training=training
        )

    def compute_output_spec(self, input_spec, **kwargs):
        """Compute output shape for symbolic tracing."""
        input_features_spec = input_spec["input_features"]
        batch_size = input_features_spec.shape[0]
        seq_len = (
            input_features_spec.shape[1] // 8
            if input_features_spec.shape[1]
            else None
        )
        return keras.KerasTensor(
            shape=(batch_size, seq_len, self.output_dim),
            dtype=input_features_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mel_bins": self.num_mel_bins,
                "d_model": self.d_model,
                "encoder_layers": self.encoder_layers_count,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "output_dim": self.output_dim,
                "downsample_hidden_size": self.downsample_hidden_size,
                "max_source_positions": self.max_source_positions,
                "scale_embedding": self.scale_embedding,
                "activation_function": self.activation_function,
                "dropout": self.dropout_rate,
            }
        )
        return config


class Qwen3OmniAudioEncoderLayer(layers.Layer):
    """Audio encoder transformer layer for Qwen3-Omni.

    A pre-norm transformer encoder layer with multi-head self-attention
    and a feed-forward network.

    Args:
        embed_dim: int. The embedding dimension (d_model).
        num_heads: int. The number of attention heads.
        ffn_dim: int. The dimension of the feed-forward network.
        activation: string. The activation function name. Defaults to `"gelu"`.
        dropout: float. The dropout rate. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        activation="gelu",
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            dtype=dtype,
            name="self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            dtype=dtype,
            name="self_attn_layer_norm",
        )

        self.fc1 = layers.Dense(
            ffn_dim,
            dtype=dtype,
            name="fc1",
        )
        self.activation_fn = layers.Activation(activation, dtype=dtype)
        self.fc2 = layers.Dense(
            embed_dim,
            dtype=dtype,
            name="fc2",
        )
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            dtype=dtype,
            name="final_layer_norm",
        )

        self.dropout_layer = layers.Dropout(dropout, dtype=dtype)

    def build(self, input_shape):
        self.self_attn.build(input_shape, input_shape)
        self.self_attn_layer_norm.build(input_shape)
        self.fc1.build(input_shape)
        self.fc2.build((input_shape[0], input_shape[1], self.ffn_dim))
        self.final_layer_norm.build(input_shape)
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=False,
    ):
        """Forward pass of the audio encoder layer.

        Args:
            hidden_states: Tensor. The input hidden states with shape
                `(batch_size, sequence_length, embed_dim)`.
            attention_mask: Tensor or None. The attention mask.
            training: bool. Whether the layer is in training mode.

        Returns:
            Tensor with shape `(batch_size, sequence_length, embed_dim)`.
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            query=hidden_states,
            value=hidden_states,
            key=hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = residual + hidden_states

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "activation": self.activation_fn.get_config()["activation"],
                "dropout": self.dropout_layer.rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
