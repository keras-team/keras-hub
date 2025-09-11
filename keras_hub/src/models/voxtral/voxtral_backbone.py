import tensorflow as tf
from keras import initializers
from keras import layers
from keras import mixed_precision

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


def voxtral_kernel_initializer(stddev=0.02):
    """Initializer for VoxTral layers (TruncatedNormal)."""
    return initializers.TruncatedNormal(stddev=stddev)


class ChunkAndPad(layers.Layer):
    """Pads and splits spectrogram into fixed-length chunks."""

    def __init__(self, frames_per_chunk, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        pad_len = (-T) % self.frames_per_chunk
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        n_chunks = tf.math.floordiv(T + pad_len, self.frames_per_chunk)
        return tf.reshape(
            x, [B * n_chunks, self.frames_per_chunk, tf.shape(x)[2]]
        )


class PositionalEmbedding(layers.Layer):
    """Learnable positional embedding per chunk."""

    def __init__(self, length, dim, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.dim = int(dim)

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.length, self.dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb[None, :, :]


class ReassembleChunks(layers.Layer):
    """Reassembles chunked outputs back into (B, T, H)."""

    def __init__(self, frames_per_chunk, postproc_chunk_len=None, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)
        self.postproc_chunk_len = postproc_chunk_len

    def call(self, processed_chunks, orig_spectrogram):
        B, T = tf.shape(orig_spectrogram)[0], tf.shape(orig_spectrogram)[1]
        n_chunks = tf.cast(
            tf.math.floordiv(
                T + self.frames_per_chunk - 1, self.frames_per_chunk
            ),
            tf.int32,
        )
        T_chunk, H = (
            tf.shape(processed_chunks)[1],
            tf.shape(processed_chunks)[2],
        )
        return tf.reshape(processed_chunks, [B, n_chunks * T_chunk, H])


@keras_hub_export("keras_hub.models.VoxTralBackbone")
class VoxTralBackbone(Backbone):
    """VoxTral audio encoder + adapter backbone."""

    def __init__(
        self,
        num_layers=32,
        num_heads=20,
        hidden_dim=1280,
        intermediate_dim=5120,
        adapter_downsample=4,
        dropout=0.1,
        max_chunk_seconds=30,
        sr=16000,
        hop_length=160,
        dtype="float32",
        **kwargs,
    ):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.adapter_downsample = int(adapter_downsample)
        self.dropout = float(dropout)
        self.max_chunk_seconds = int(max_chunk_seconds)
        self.sr = int(sr)
        self.hop_length = int(hop_length)

        # Frames per chunk before conv
        self.frames_per_chunk_preconv = int(
            self.max_chunk_seconds * (self.sr / self.hop_length)
        )
        self.postconv_frames_per_chunk = self.frames_per_chunk_preconv // 2

        # Determine layer dtype for mixed precision
        if isinstance(dtype, mixed_precision.Policy):
            self.layer_dtype = dtype.compute_dtype
        else:
            self.layer_dtype = dtype

        # Conv1D stem
        self.conv_stem_1 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=self.layer_dtype,
            name="conv_stem_1",
        )
        self.conv_stem_2 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=self.layer_dtype,
            name="conv_stem_2",
        )

        # Transformer layers
        self.transformer_layers = [
            TransformerEncoder(
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                name=f"transformer_layer_{i}",
            )
            for i in range(self.num_layers)
        ]

        # Adapter
        self.adapter_dense = layers.Dense(
            self.hidden_dim,
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=self.layer_dtype,
            name="adapter_dense",
        )
        self.adapter_pool = layers.AveragePooling1D(
            pool_size=self.adapter_downsample,
            strides=self.adapter_downsample,
            padding="valid",
            name="adapter_downsample",
        )

        # Positional embeddings
        self.pos_emb = PositionalEmbedding(
            self.postconv_frames_per_chunk, self.hidden_dim, name="pos_emb"
        )

        # Functional model
        spectrogram_input = tf.keras.Input(
            shape=(None, 128), dtype=self.layer_dtype, name="spectrogram"
        )
        x = ChunkAndPad(self.frames_per_chunk_preconv, name="chunk_and_pad")(
            spectrogram_input
        )
        x = self.conv_stem_1(x)
        x = self.conv_stem_2(x)
        x = self.pos_emb(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.adapter_dense(x)
        x = self.adapter_pool(x)
        outputs = ReassembleChunks(
            self.frames_per_chunk_preconv, name="reassemble_chunks"
        )(x, spectrogram_input)

        super().__init__(
            inputs=spectrogram_input, outputs=outputs, dtype=dtype, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "adapter_downsample": self.adapter_downsample,
                "dropout": self.dropout,
                "max_chunk_seconds": self.max_chunk_seconds,
                "sr": self.sr,
                "hop_length": self.hop_length,
            }
        )
        return config
