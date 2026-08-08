import keras
from keras import ops

from keras_hub.src import api_export
from keras_hub.src.layers.modeling.sine_position_encoding import (
    SinePositionEncoding,
)
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder


def _post_cnn_length(lengths):
    """Length after three (k=3, s=2, p=1) convolutions."""
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
        output_dim: int. Output dimension of the projector.
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
        self._conv2d1 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d1",
        )
        self._conv2d2 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d2",
        )
        self._conv2d3 = keras.layers.Conv2D(
            downsample_hidden_size,
            3,
            strides=2,
            padding="same",
            activation="gelu",
            name="conv2d3",
        )

        # Projection
        self._conv_out = keras.layers.Dense(
            d_model, use_bias=False, name="conv_out"
        )

        # Position Encoding
        self._position_encoding = SinePositionEncoding(
            name="positional_embedding"
        )

        # Transformer Layers
        self._transformer_layers = []
        for i in range(encoder_layers):
            layer = TransformerEncoder(
                intermediate_dim=encoder_ffn_dim,
                num_heads=encoder_attention_heads,
                dropout=dropout,
                activation=activation_function,
                name=f"transformer_layer_{i}",
            )
            self._transformer_layers.append(layer)

        self._ln_post = keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_post"
        )

        # Projector
        self._proj_linear_1 = keras.layers.Dense(
            d_model, activation=activation_function, name="proj_linear_1"
        )
        self._proj_linear_2 = keras.layers.Dense(
            output_dim, name="proj_linear_2"
        )
        self.built = True

    def get_num_audio_tokens(self, audio_mel_shape):
        """Calculate number of audio tokens for a given mel spectrogram shape."""
        # Based on 8x downsampling (3 stride-2 layers)
        # T frames -> ceil(T / 8) tokens.
        # However, for chunk-based processing, each 100-frame chunk produces
        # exactly 13 tokens (as confirmed by technical report).
        # We use the _post_cnn_length logic here.
        T = audio_mel_shape[1]
        chunk_len = self.chunk_len
        num_chunks = T // chunk_len
        return num_chunks * 13

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

        x = self._conv2d1(audio_mel)
        x = self._conv2d2(x)
        x = self._conv2d3(x)

        W_out = ops.shape(x)[2]
        x = ops.transpose(x, (0, 2, 3, 1))
        x = ops.reshape(x, (-1, W_out, x.shape[2] * x.shape[3]))
        x = self._conv_out(x)

        # Positional Encoding
        x = x + self._position_encoding(x)

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

        for transformer_layer in self._transformer_layers:
            x = transformer_layer(
                x, padding_mask=padding_mask, attention_mask=attention_mask
            )

        x = self._ln_post(x)
        x = self._proj_linear_1(x)
        x = self._proj_linear_2(x)
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
