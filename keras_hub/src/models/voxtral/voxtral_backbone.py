from keras import Input
from keras import initializers
from keras import layers
from keras import mixed_precision
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


def voxtral_kernel_initializer(stddev=0.02):
    """
    Create a TruncatedNormal initializer for VoxTral layers.

    Args:
        stddev (float): Standard deviation of the truncated normal distribution.

    Returns:
        keras.initializers.Initializer: Truncated normal initializer.
    """
    return initializers.TruncatedNormal(stddev=stddev)


class ChunkAndPad(layers.Layer):
    """
    Pads and splits an input spectrogram into fixed-length chunks.

    This layer ensures the time axis of the input is divisible by
    `frames_per_chunk` by padding zeros, then reshapes into
    `(batch * n_chunks, frames_per_chunk, features)`.

    Args:
        frames_per_chunk (int): Number of frames per chunk.
    """

    def __init__(self, frames_per_chunk, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)

    def call(self, x):
        """Pad and chunk the input tensor along time dimension."""
        B, T = ops.shape(x)[0], ops.shape(x)[1]
        pad_len = (-T) % self.frames_per_chunk
        x = ops.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        n_chunks = ops.floor_divide(T + pad_len, self.frames_per_chunk)
        return ops.reshape(
            x, [B * n_chunks, self.frames_per_chunk, ops.shape(x)[2]]
        )

    def compute_output_shape(self, input_shape):
        """
        Compute static output shape for Keras/JAX backends.

        Args:
            input_shape (tuple): (batch, time, features).

        Returns:
            tuple: (batch * n_chunks, frames_per_chunk, features)
        """
        batch, time, feat = input_shape
        if time is None:
            n_chunks = None
        else:
            import math

            n_chunks = math.ceil(time / self.frames_per_chunk)
        return (
            None if batch is None else batch * n_chunks,
            self.frames_per_chunk,
            feat,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"frames_per_chunk": self.frames_per_chunk})
        return config


class PositionalEmbedding(layers.Layer):
    """
    Learnable positional embedding added to each time step in a chunk.

    Args:
        length (int): Sequence length of each chunk
        (frames per chunk post-conv).
        dim (int): Embedding dimension.
    """

    def __init__(self, length, dim, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.dim = int(dim)

    def build(self, input_shape):
        """Create the embedding weights."""
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.length, self.dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=self.compute_dtype,
        )
        super().build(input_shape)

    def call(self, x):
        """
        Add the positional embedding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch, time, dim).

        Returns:
            Tensor: Input tensor with positional embedding added.
        """
        return x + ops.cast(self.pos_emb[None, :, :], x.dtype)

    def compute_output_shape(self, input_shape):
        """Return same shape as input."""
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"length": self.length, "dim": self.dim})
        return config


class ReassembleChunks(layers.Layer):
    """
    Reassembles chunked outputs back into `(batch, time, hidden_dim)`.

    Args:
        frames_per_chunk (int): Frames per chunk pre-conv.
        postproc_chunk_len (int, optional): Chunk length after processing.
    """

    def __init__(self, frames_per_chunk, postproc_chunk_len=None, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_chunk = int(frames_per_chunk)
        self.postproc_chunk_len = (
            None if postproc_chunk_len is None else int(postproc_chunk_len)
        )

    def call(self, processed_chunks, orig_spectrogram):
        """
        Reassemble processed chunks into a continuous time sequence.

        Args:
            processed_chunks (Tensor): Output of transformer layers
                of shape (B*n_chunks, T_chunk, H).
            orig_spectrogram (Tensor): Original input spectrogram
                of shape (B, T, F).

        Returns:
            Tensor: Reassembled tensor of shape (B, T', H).
        """
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

    def compute_output_shape(self, input_shape):
        """Return shape compatible with a single long sequence."""
        return input_shape

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
    """
    VoxTral audio encoder + adapter backbone.

    This model implements the encoder portion of the VoxTral model.
    It takes a log-Mel spectrogram and produces a sequence of hidden states.

    Args:
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Embedding size.
        intermediate_dim (int): Size of feedforward network hidden layer.
        adapter_downsample (int): Pooling factor after adapter dense.
        dropout (float): Dropout probability.
        max_chunk_seconds (int): Chunking length in seconds.
        sr (int): Audio sample rate.
        hop_length (int): Hop length for spectrogram frames.
        dtype (str or mixed_precision.Policy, optional): Layer dtype.
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
        """Initialize the VoxTral backbone."""
        # Store configuration
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

        # --- Mixed precision policy ---
        if dtype is None:
            policy = mixed_precision.global_policy()
        elif isinstance(dtype, str):
            policy = mixed_precision.Policy(dtype)
        elif isinstance(dtype, dict):  # coming from config
            policy = mixed_precision.Policy(dtype["config"]["name"])
        else:
            policy = dtype  # already a Policy

        variable_dtype = policy.variable_dtype
        compute_dtype = policy.compute_dtype
        self._policy = policy  # save for get_config()

        # --- Layers ---
        self.conv_stem_1 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=variable_dtype,
            name="conv_stem_1",
        )
        self.conv_stem_2 = layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=variable_dtype,
            name="conv_stem_2",
        )

        self.transformer_layers = [
            TransformerEncoder(
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                name=f"transformer_layer_{i}",
                dtype=variable_dtype,
            )
            for i in range(self.num_layers)
        ]

        self.adapter_dense = layers.Dense(
            self.hidden_dim,
            activation="relu",
            kernel_initializer=voxtral_kernel_initializer(),
            dtype=variable_dtype,
            name="adapter_dense",
        )
        self.adapter_pool = layers.AveragePooling1D(
            pool_size=self.adapter_downsample,
            strides=self.adapter_downsample,
            padding="valid",
            name="adapter_downsample",
            dtype=variable_dtype,
        )

        self.pos_emb = PositionalEmbedding(
            self.postconv_frames_per_chunk,
            self.hidden_dim,
            name="pos_emb",
            dtype=variable_dtype,
        )

        # --- Functional graph ---
        spectrogram_input = Input(
            shape=(None, 128), dtype=compute_dtype, name="spectrogram"
        )
        x = ChunkAndPad(
            self.frames_per_chunk_preconv,
            name="chunk_and_pad",
            dtype=compute_dtype,
        )(spectrogram_input)
        x = self.conv_stem_1(x)
        x = self.conv_stem_2(x)
        x = self.pos_emb(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.adapter_dense(x)
        x = self.adapter_pool(x)
        outputs = ReassembleChunks(
            self.frames_per_chunk_preconv,
            name="reassemble_chunks",
            dtype=compute_dtype,
        )(x, spectrogram_input)

        super().__init__(
            inputs=spectrogram_input,
            outputs=outputs,
            dtype=compute_dtype,
            **kwargs,
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
                "dtype": self._policy.name,  # store string
            }
        )
        return config
