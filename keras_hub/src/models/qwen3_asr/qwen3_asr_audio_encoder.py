import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder


def compute_sinusoidal_positional_embedding(
    length, channels, max_timescale=10000
):
    if channels % 2 != 0:
        raise ValueError("channels needs to be even")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(
        -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
    )
    scaled_time = (
        np.arange(length, dtype=np.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )
    pos_emb = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return pos_emb


@keras_hub_export("keras_hub.models.Qwen3ASRAudioEncoder")
class Qwen3ASRAudioEncoder(keras.layers.Layer):
    """Qwen3 ASR Audio Encoder.

    This component processes the log-mel spectrogram features and encodes them
    into audio embeddings.

    Args:
        num_mel_bins: int. Number of mel bins in the input spectrogram.
            Defaults to `128`.
        num_layers: int. Number of transformer layers. Defaults to `24`.
        num_attention_heads: int. Number of attention heads. Defaults to `16`.
        intermediate_dim: int. FFN intermediate dimension. Defaults to `4096`.
        d_model: int. Hidden dimension of the encoder. Defaults to `1024`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        n_window: int. Half the chunk size. Defaults to `50`.
        n_window_infer: int. Inference window size. Defaults to `800`.
        downsample_hidden_size: int. Hidden size of conv stem.
            Defaults to `480`.
        max_position_embeddings: int. Max position embeddings for chunk.
            Defaults to `13`.
    """

    def __init__(
        self,
        num_mel_bins=128,
        num_layers=24,
        num_attention_heads=16,
        intermediate_dim=4096,
        d_model=1024,
        dropout=0.0,
        n_window=50,
        n_window_infer=800,
        downsample_hidden_size=480,
        max_position_embeddings=13,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.d_model = d_model
        self.dropout = dropout
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.downsample_hidden_size = downsample_hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.chunk_len = n_window * 2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mel_bins": self.num_mel_bins,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_dim": self.intermediate_dim,
                "d_model": self.d_model,
                "dropout": self.dropout,
                "n_window": self.n_window,
                "n_window_infer": self.n_window_infer,
                "downsample_hidden_size": self.downsample_hidden_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config

    def build(self, input_shape=None, **kwargs):
        self.padding = keras.layers.ZeroPadding2D(
            padding=1,
            dtype=self.dtype_policy,
        )
        self.conv2d1 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            activation="gelu",
            dtype=self.dtype_policy,
            name="conv2d1",
        )
        self.conv2d2 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            activation="gelu",
            dtype=self.dtype_policy,
            name="conv2d2",
        )
        self.conv2d3 = keras.layers.Conv2D(
            self.downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            activation="gelu",
            dtype=self.dtype_policy,
            name="conv2d3",
        )

        # Calculate conv_out_dim
        freq_bins = self.num_mel_bins
        for _ in range(3):
            freq_bins = (freq_bins + 1) // 2
        conv_out_dim = self.downsample_hidden_size * freq_bins

        self.conv_out = keras.layers.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype_policy,
            name="conv_out",
        )

        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(self.max_position_embeddings, self.d_model),
            initializer=keras.initializers.Constant(
                compute_sinusoidal_positional_embedding(
                    self.max_position_embeddings, self.d_model
                )
            ),
            trainable=False,
            dtype=self.variable_dtype,
        )

        self.transformer_layers = [
            TransformerEncoder(
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_attention_heads,
                dropout=self.dropout,
                activation="gelu",
                normalize_first=True,
                dtype=self.dtype_policy,
                name=f"transformer_layer_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.ln_post = keras.layers.LayerNormalization(
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="ln_post",
        )

        # Build sub-layers
        self.conv2d1.build((None, None, None, 1))
        self.conv2d2.build((None, None, None, self.downsample_hidden_size))
        self.conv2d3.build((None, None, None, self.downsample_hidden_size))
        self.conv_out.build((None, None, conv_out_dim))
        for layer in self.transformer_layers:
            layer.build((None, None, self.d_model))
        self.ln_post.build((None, None, self.d_model))

        super().build(input_shape)

    def _post_cnn_length(self, lengths):
        # Length after three (k=3, s=2, p=1) convolutions
        for _ in range(3):
            lengths = ops.where(
                lengths > 0,
                (lengths - 1) // 2 + 1,
                ops.zeros_like(lengths),
            )
        return lengths

    def call(self, input_features, input_features_mask):
        # input_features shape: (B, T, num_mel_bins)
        # input_features_mask shape: (B, T)

        # Transpose to (B, num_mel_bins, T) first
        x = ops.transpose(input_features, (0, 2, 1))  # (B, num_mel_bins, T)

        batch_size = ops.shape(x)[0]
        T = ops.shape(x)[2]

        # Pad T to multiple of chunk_len if necessary
        # (mostly for dummy shapes during build)
        pad_len = (self.chunk_len - (T % self.chunk_len)) % self.chunk_len
        x = ops.pad(x, [[0, 0], [0, 0], [0, pad_len]])
        input_features_mask = ops.pad(
            input_features_mask, [[0, 0], [0, pad_len]]
        )

        # Chunk and process through CNN
        chunked = ops.reshape(
            x, (batch_size, self.num_mel_bins, -1, self.chunk_len)
        )
        chunked = ops.transpose(
            chunked, (0, 2, 1, 3)
        )  # (B, num_chunks, num_mel_bins, chunk_len)
        num_chunks = ops.shape(chunked)[1]

        chunked = ops.reshape(
            chunked,
            (-1, self.num_mel_bins, self.chunk_len, 1),
        )

        conv_out = ops.pad(chunked, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d1(conv_out)
        conv_out = ops.pad(conv_out, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d2(conv_out)
        conv_out = ops.pad(conv_out, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d3(
            conv_out
        )  # (B * num_chunks, freq_bins, time_steps, downsample_hidden_size)

        shape = ops.shape(conv_out)
        freq_bins = shape[1]
        time_steps = shape[2]
        conv_channels = shape[3]

        conv_out = ops.transpose(
            conv_out, (0, 2, 1, 3)
        )  # (B * num_chunks, time_steps, freq_bins, C)
        conv_out = ops.reshape(
            conv_out,
            (batch_size * num_chunks, time_steps, freq_bins * conv_channels),
        )

        conv_out = self.conv_out(
            conv_out
        )  # (B * num_chunks, time_steps, d_model)

        # Add positional embedding
        # Cast to compute_dtype to support mixed precision
        positional_embedding = ops.cast(
            self.positional_embedding, self.compute_dtype
        )
        conv_out = conv_out + positional_embedding[:time_steps, :]

        # Reshape back to (B, num_chunks * time_steps, d_model)
        conv_out = ops.reshape(conv_out, (batch_size, -1, self.d_model))

        # Compute mask
        chunk_masks = ops.reshape(
            input_features_mask, (batch_size, -1, self.chunk_len)
        )
        chunk_lens = ops.sum(chunk_masks, axis=-1)
        chunk_aftercnn_lens = self._post_cnn_length(chunk_lens)
        aftercnn_lens = ops.sum(chunk_aftercnn_lens, axis=-1)
        total_len = ops.shape(conv_out)[1]

        # Construct block-diagonal mask
        n_window_ratio = self.n_window_infer // (self.n_window * 2)
        window_aftercnn = time_steps * n_window_ratio

        arange = ops.arange(total_len, dtype="int32")
        arange = ops.expand_dims(arange, axis=0)  # (1, total_len)
        aftercnn_lens_ex = ops.expand_dims(aftercnn_lens, axis=-1)  # (B, 1)

        window_ids = ops.where(
            arange < aftercnn_lens_ex, arange // window_aftercnn, -1
        )

        window_ids_i = ops.expand_dims(window_ids, axis=2)  # (B, L, 1)
        window_ids_j = ops.expand_dims(window_ids, axis=1)  # (B, 1, L)
        attention_mask = (
            (window_ids_i == window_ids_j)
            & (window_ids_i != -1)
            & (window_ids_j != -1)
        )

        # Run transformer layers
        hidden_states = conv_out
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.ln_post(hidden_states)
        return hidden_states


@keras_hub_export("keras_hub.models.Qwen3ASRMultiModalProjector")
class Qwen3ASRMultiModalProjector(keras.layers.Layer):
    """Qwen3 ASR MultiModal Projector.

    Projects audio encoder features to the LLM input dimension.
    """

    def __init__(self, output_dim, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "activation": self.activation,
            }
        )
        return config

    def build(self, input_shape=None, **kwargs):
        if input_shape is not None:
            d_model = input_shape[-1]
        else:
            # Fallback if called with kwargs
            d_model = kwargs.get("input_features_shape", [None, None, 32])[
                -1
            ]  # fallback default

        self.linear_1 = keras.layers.Dense(
            d_model,
            dtype=self.dtype_policy,
            name="linear_1",
        )
        self.act = keras.layers.Activation(
            self.activation,
            dtype=self.dtype_policy,
            name="act",
        )
        self.linear_2 = keras.layers.Dense(
            self.output_dim,
            dtype=self.dtype_policy,
            name="linear_2",
        )

        self.linear_1.build((None, None, d_model))
        self.linear_2.build((None, None, d_model))

        super().build(input_shape)

    def call(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
