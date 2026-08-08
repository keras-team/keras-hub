import math

import keras
from keras import ops

from keras_hub.src import api_export
from keras_hub.src.layers.modeling import transformer_encoder


class SinusoidsPositionEmbedding(keras.layers.Layer):
    """Sinusoidal position embedding for Qwen3-ASR."""

    def __init__(self, length, channels, max_timescale=10000, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError(
                "SinusoidsPositionEmbedding needs even channels input"
            )

        log_timescale_increment = math.log(self.max_timescale) / (
            self.channels // 2 - 1
        )
        inv_timescales = ops.exp(
            -log_timescale_increment
            * ops.cast(ops.arange(self.channels // 2), "float32")
        )

        # We can't easily compute this statically if length is dynamic,
        # but usually length is max_position_embeddings.

        scaled_time = ops.expand_dims(
            ops.cast(ops.arange(self.length), "float32"), axis=1
        ) * ops.expand_dims(inv_timescales, axis=0)
        positional_embedding = ops.concatenate(
            [
                ops.sin(scaled_time),
                ops.cos(ops.convert_to_tensor(scaled_time)),
            ],
            axis=1,
        )

        self.positional_embedding = ops.cast(positional_embedding, "float32")

    def call(self, seqlen):
        return ops.slice(
            self.positional_embedding, [0, 0], [seqlen, self.channels]
        )


def _post_cnn_length(lengths):
    """Length after three (k=3, s=2, p=1) convolutions."""
    # In Keras, we can use ops.where and integer division.
    # But wait, lengths might be a tensor of counts.
    for _ in range(3):
        lengths = ops.where(
            lengths > 0, (lengths - 1) // 2 + 1, ops.zeros_like(lengths)
        )
    return lengths


@api_export.keras_hub_export("keras_hub.models.Qwen3ASRAudioEncoder")
class Qwen3ASRAudioEncoder(keras.Model):
    """Qwen3-ASR Audio Encoder.

    Consists of 3 Conv2D layers followed by Transformer Encoder layers.
    Processes audio in chunks independently in CNN.

    Args:
        d_model: int. Hidden dimension of the transformer layers.
        encoder_layers: int. Number of transformer layers.
        encoder_attention_heads: int. Number of attention heads.
        encoder_ffn_dim: int. Hidden dimension of the feed-forward network.
        downsample_hidden_size: int. Hidden dimension of downsampling CNN.
        num_mel_bins: int. Number of mel bins in input.
        n_window: int. Half the chunk size.
        max_position_embeddings: int. Maximum position embeddings.
        dropout: float. Dropout rate.
        attention_dropout: float. Attention dropout rate.
        activation_function: str. Activation function in MLP.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        d_model=768,
        encoder_layers=24,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        downsample_hidden_size=512,
        num_mel_bins=128,
        n_window=50,
        max_position_embeddings=1500,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        output_dim=3584,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.num_mel_bins = num_mel_bins
        self.n_window = n_window
        self.max_position_embeddings = max_position_embeddings
        self.output_dim = output_dim

        self.chunk_len = n_window * 2

        # CNN Layers
        self.conv2d1 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d1",
        )
        self.conv2d2 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d2",
        )
        self.conv2d3 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d3",
        )

        # Projection
        # Output of CNN has 16 freq bins (128 // 8).
        self.freq_bins_out = 16
        self.conv_out = keras.layers.Dense(
            d_model, use_bias=False, name="conv_out"
        )

        # Position Embedding
        # Output sequence length of a chunk in time dimension is 13 (100 // 8 + 1).
        self.chunk_time_steps_out = 13
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_position_embeddings, d_model, name="positional_embedding"
        )

        # Transformer Layers
        self.transformer_layers = []
        for i in range(encoder_layers):
            layer = transformer_encoder.TransformerEncoder(
                intermediate_dim=encoder_ffn_dim,
                num_heads=encoder_attention_heads,
                dropout=dropout,
                activation=activation_function,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.ln_post = keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_post"
        )

        # Projector
        self.proj_linear_1 = keras.layers.Dense(
            d_model, activation=activation_function, name="proj_linear_1"
        )
        self.proj_linear_2 = keras.layers.Dense(
            output_dim, name="proj_linear_2"
        )
        self.built = True

    def _build_attention_mask(self, batch_size, num_chunks, time_steps):
        # Build block diagonal attention mask to block attention across chunks
        # We can create a mask of shape (total_steps, total_steps)
        # where elements are True only if they belong to the same chunk.
        # This is static for a given batch shape if we don't pack.
        # Let's try to do it dynamically.
        # Chunk indices for each position:
        chunk_indices = ops.repeat(ops.arange(num_chunks), repeats=time_steps)
        # Compare indices:
        chunk_indices_exp1 = ops.expand_dims(chunk_indices, axis=-1)
        chunk_indices_exp2 = ops.expand_dims(chunk_indices, axis=0)
        # Mask: (total_steps, total_steps)
        mask = ops.equal(chunk_indices_exp1, chunk_indices_exp2)

        # Expand to (B, total_steps, total_steps)
        mask = ops.expand_dims(mask, axis=0)
        mask = ops.repeat(mask, repeats=batch_size, axis=0)

        return mask

    def call(self, audio_mel, audio_mel_mask=None):
        """Encode audio mel features.

        Args:
            audio_mel: Tensor of shape ``(B, T, num_mel_bins)``.
            audio_mel_mask: Tensor of shape ``(B, T)``. True = valid.

        Returns:
            Tensor of shape ``(B, T_out, d_model)``.
        """
        B = ops.shape(audio_mel)[0]
        T = ops.shape(audio_mel)[1]
        F = ops.shape(audio_mel)[2]

        num_chunks = T // self.chunk_len

        # Chunking time dimension
        # (B, T, F) -> (B, num_chunks, chunk_len, F)
        audio_mel = ops.reshape(audio_mel, (B, -1, self.chunk_len, F))

        # Prepare for 2D Conv: (B_packed, H, W, C) where H=Freq, W=Time, C=1
        # (B, num_chunks, chunk_len, F) -> (B, num_chunks, F, chunk_len)
        audio_mel = ops.transpose(audio_mel, (0, 1, 3, 2))
        # Reshape to (B * num_chunks, F, chunk_len, 1)
        audio_mel = ops.reshape(audio_mel, (-1, F, self.chunk_len, 1))

        # CNN Layers
        x = self.conv2d1(audio_mel)
        x = self.conv2d2(x)
        x = self.conv2d3(x)

        # Output shape: (B * num_chunks, H_out, W_out, downsample_hidden_size)
        # Where H_out = 16, W_out = 13 (for chunk_len=100)
        W_out = ops.shape(x)[2]

        # Permute to (B * num_chunks, W_out, downsample_hidden_size, H_out)
        # Index: 0, 1, 2, 3 -> B_packed, H, W, C
        # We want (B_packed, W, C, H) -> 0, 2, 3, 1
        x = ops.transpose(x, (0, 2, 3, 1))

        # Flatten last two: (B * num_chunks, W_out, downsample_hidden_size * H_out)
        x = ops.reshape(x, (-1, W_out, x.shape[2] * x.shape[3]))

        # Projection
        x = self.conv_out(x)

        # Positional Embedding
        # Added to each chunk locally
        pos_emb = self.positional_embedding(seqlen=W_out)
        # Expand pos_emb to match batch dim: (1, W_out, D)
        pos_emb = ops.expand_dims(pos_emb, axis=0)
        x = x + pos_emb

        # Reshape back to (B, num_chunks * W_out, d_model)
        x = ops.reshape(x, (B, -1, self.d_model))

        # Update Mask if provided
        padding_mask = None
        attention_mask = None
        if audio_mel_mask is not None:
            # Reshape input mask: (B, T) -> (B, num_chunks, chunk_len)
            mask_chunked = ops.reshape(audio_mel_mask, (B, -1, self.chunk_len))
            # Sum valid elements per chunk
            chunk_valid_lens = ops.sum(ops.cast(mask_chunked, "int32"), axis=-1)
            # Compute new valid lengths
            valid_lens_after_cnn = _post_cnn_length(chunk_valid_lens)

            # Create new padding mask of shape (B, num_chunks, W_out)
            # This is slightly tricky, but we can do it with comparison.
            indices = ops.arange(W_out)
            indices = ops.expand_dims(indices, axis=0)
            indices = ops.expand_dims(indices, axis=0)  # (1, 1, W_out)

            valid_lens_after_cnn_exp = ops.expand_dims(
                valid_lens_after_cnn, axis=-1
            )  # (B, num_chunks, 1)

            new_mask = ops.less(
                indices, valid_lens_after_cnn_exp
            )  # (B, num_chunks, W_out)

            # Reshape to (B, num_chunks * W_out)
            padding_mask = ops.reshape(new_mask, (B, -1))

            # Build attention mask to block across chunks
            attention_mask = self._build_attention_mask(B, num_chunks, W_out)

            # Combine attention_mask with padding_mask?
            # TransformerEncoder usually handles padding_mask separately, but customized attention_mask can override it.
            # If we provide attention_mask, we should ensure it also handles padding.
            # Let's check TransformerEncoder docstring.
            # It says attention_mask overrides if provided.
            # So attention_mask should also block padding!
            # padding_mask_exp1 = ops.expand_dims(padding_mask, axis=-1)
            # padding_mask_exp2 = ops.expand_dims(padding_mask, axis=1)
            # padding_mask_2d = ops.logical_and(padding_mask_exp1, padding_mask_exp2)
            # attention_mask = ops.logical_and(attention_mask, padding_mask_2d)

        # Transformer Layers
        for transformer_layer in self.transformer_layers:
            # Pass custom attention mask if available
            # Wait, TransformerEncoder might not expect 3D attention_mask in this shape?
            # It expects [batch_size, sequence_length, sequence_length].
            # Our attention_mask is (B, total_steps, total_steps).
            # This matches PERFECTLY.
            # But let's verify if attention_mask is preferred over padding_mask.
            x = transformer_layer(
                x, padding_mask=padding_mask, attention_mask=attention_mask
            )

        x = self.ln_post(x)

        # Apply Projector
        x = self.proj_linear_1(x)
        x = self.proj_linear_2(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "encoder_layers": self.encoder_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "downsample_hidden_size": self.downsample_hidden_size,
                "num_mel_bins": self.num_mel_bins,
                "n_window": self.n_window,
                "max_position_embeddings": self.max_position_embeddings,
                "output_dim": self.output_dim,
            }
        )
        return config
