import keras
from keras import ops

from keras_hub.src import api_export
from keras_hub.src.layers.modeling import transformer_encoder


def _post_cnn_length(lengths):
    """Length after three (k=3, s=2, p=1) convolutions."""
    for _ in range(3):
        lengths = ops.where(
            lengths > 0, (lengths - 1) // 2 + 1, ops.zeros_like(lengths)
        )
    return lengths


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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "length": self.length,
                "channels": self.channels,
                "max_timescale": self.max_timescale,
            }
        )
        return config


import math


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
        # Output sequence length of a chunk in time dimension is 13
        # (100 // 8 + 1).
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
        chunk_indices = ops.repeat(ops.arange(num_chunks), repeats=time_steps)
        chunk_indices_exp1 = ops.expand_dims(chunk_indices, axis=-1)
        chunk_indices_exp2 = ops.expand_dims(chunk_indices, axis=0)
        mask = ops.equal(chunk_indices_exp1, chunk_indices_exp2)
        mask = ops.expand_dims(mask, axis=0)
        mask = ops.repeat(mask, repeats=batch_size, axis=0)
        return mask

    def call(self, audio_mel, audio_mel_mask=None):
        B = ops.shape(audio_mel)[0]
        T = ops.shape(audio_mel)[1]
        F = ops.shape(audio_mel)[2]

        num_chunks = T // self.chunk_len
        audio_mel = ops.reshape(audio_mel, (B, -1, self.chunk_len, F))
        audio_mel = ops.transpose(audio_mel, (0, 1, 3, 2))
        audio_mel = ops.reshape(audio_mel, (-1, F, self.chunk_len, 1))

        x = self.conv2d1(audio_mel)
        x = self.conv2d2(x)
        x = self.conv2d3(x)

        W_out = ops.shape(x)[2]
        x = ops.transpose(x, (0, 2, 3, 1))
        x = ops.reshape(x, (-1, W_out, x.shape[2] * x.shape[3]))
        x = self.conv_out(x)

        pos_emb = self.positional_embedding(seqlen=W_out)
        pos_emb = ops.expand_dims(pos_emb, axis=0)
        x = x + pos_emb

        x = ops.reshape(x, (B, -1, self.d_model))

        padding_mask = None
        attention_mask = None
        if audio_mel_mask is not None:
            mask_chunked = ops.reshape(audio_mel_mask, (B, -1, self.chunk_len))
            chunk_valid_lens = ops.sum(ops.cast(mask_chunked, "int32"), axis=-1)
            valid_lens_after_cnn = _post_cnn_length(chunk_valid_lens)

            indices = ops.arange(W_out)
            indices = ops.expand_dims(indices, axis=0)
            indices = ops.expand_dims(indices, axis=0)
            valid_lens_after_cnn_exp = ops.expand_dims(
                valid_lens_after_cnn, axis=-1
            )
            new_mask = ops.less(indices, valid_lens_after_cnn_exp)
            padding_mask = ops.reshape(new_mask, (B, -1))
            attention_mask = self._build_attention_mask(B, num_chunks, W_out)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x, padding_mask=padding_mask, attention_mask=attention_mask
            )

        x = self.ln_post(x)
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
