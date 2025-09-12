from keras import Input
from keras import initializers
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


def voxtral_kernel_initializer(stddev=0.02):
    """Initializer for VoxTral layers (TruncatedNormal)."""
    return initializers.TruncatedNormal(stddev=stddev)


class ChunkAndPad(layers.Layer):
    """Pads and splits spectrogram into fixed-length chunks.

    Args:
        frames_per_chunk: int. Number of frames per chunk.
    """

    def __init__(self, frames_per_chunk, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)

    def call(self, x):
        B, T = ops.shape(x)[0], ops.shape(x)[1]
        pad_len = (-T) % self.frames_per_chunk
        x = ops.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        n_chunks = ops.floor_divide(T + pad_len, self.frames_per_chunk)
        return ops.reshape(
            x, [B * n_chunks, self.frames_per_chunk, ops.shape(x)[2]]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"frames_per_chunk": self.frames_per_chunk})
        return config


class PositionalEmbedding(layers.Layer):
    """Learnable positional embedding per chunk.

    Args:
        length: int. Sequence length.
        dim: int. Embedding dimension.
    """

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
            dtype=self.compute_dtype,
        )
        super().build(input_shape)

    def call(self, x):
        # Cast embedding to input dtype to avoid float16/32 mismatch
        return x + ops.cast(self.pos_emb[None, :, :], x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"length": self.length, "dim": self.dim})
        return config


class ReassembleChunks(layers.Layer):
    """Reassembles chunked outputs back into (B, T, H).

    Args:
        frames_per_chunk: int. Frames per chunk pre-conv.
        postproc_chunk_len: Optional[int]. Post-processing chunk length.
    """

    def __init__(self, frames_per_chunk, postproc_chunk_len=None, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)
        self.postproc_chunk_len = (
            None if postproc_chunk_len is None else int(postproc_chunk_len)
        )

    def call(self, processed_chunks, orig_spectrogram):
        B, T = ops.shape(orig_spectrogram)[0], ops.shape(orig_spectrogram)[1]
        n_chunks = ops.cast(
            ops.floor_divide(
                T + self.frames_per_chunk - 1, self.frames_per_chunk
            ),
            "int32",
        )
        T_chunk, H = (
            ops.shape(processed_chunks)[1],
            ops.shape(processed_chunks)[2],
        )
        return ops.reshape(processed_chunks, [B, n_chunks * T_chunk, H])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "frames_per_chunk": self.frames_per_chunk,
                "postproc_chunk_len": self.postproc_chunk_len,
            }
        )
        return config


@keras_hub_export("keras_hub.models.VoxTralBackbone")
class VoxTralBackbone(Backbone):
    """VoxTral audio encoder + adapter backbone.

    This model implements the encoder portion of the VoxTral model. It takes
    a log-Mel spectrogram and produces a sequence of hidden states.

    Args:
        num_layers: int, number of transformer layers.
        num_heads: int, number of attention heads.
        hidden_dim: int, embedding size.
        intermediate_dim: int, size of feedforward network hidden layer.
        adapter_downsample: int, pooling factor after adapter dense.
        dropout: float, dropout probability.
        max_chunk_seconds: int, length of chunking in seconds.
        sr: int, sample rate.
        hop_length: int, hop length for spectrogram frames.
        dtype: str or mixed_precision.Policy, dtype for layers.

    Example:
        ```python
        from keras_hub.models import VoxTralBackbone
        model = VoxTralBackbone()
        output = model(input_tensor)
        ```
    """

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
        dtype=None,
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

        # Conv1D stem
        self.conv_stem_1 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=dtype,
            name="conv_stem_1",
        )
        self.conv_stem_2 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=dtype,
            name="conv_stem_2",
        )

        # Transformer layers
        self.transformer_layers = [
            TransformerEncoder(
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                name=f"transformer_layer_{i}",
                dtype=dtype,
            )
            for i in range(self.num_layers)
        ]

        # Adapter
        self.adapter_dense = layers.Dense(
            self.hidden_dim,
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=dtype,
            name="adapter_dense",
        )
        self.adapter_pool = layers.AveragePooling1D(
            pool_size=self.adapter_downsample,
            strides=self.adapter_downsample,
            padding="valid",
            name="adapter_downsample",
            dtype=dtype,
        )

        # Positional embeddings
        self.pos_emb = PositionalEmbedding(
            self.postconv_frames_per_chunk,
            self.hidden_dim,
            name="pos_emb",
            dtype=dtype,
        )

        # Functional model
        spectrogram_input = Input(
            shape=(None, 128), dtype="float32", name="spectrogram"
        )

        x = ChunkAndPad(
            self.frames_per_chunk_preconv, name="chunk_and_pad", dtype="float32"
        )(spectrogram_input)
        x = self.conv_stem_1(x)
        x = self.conv_stem_2(x)
        x = self.pos_emb(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.adapter_dense(x)
        x = self.adapter_pool(x)
        outputs = ReassembleChunks(
            self.frames_per_chunk_preconv, name="reassemble_chunks", dtype=dtype
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
