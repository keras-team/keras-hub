import math

import keras
import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export


def _get_feat_extract_output_length(input_length, n_window=100):
    """Helps compute static block sizes for the windowed audio encoder without
    running the torch-only utility.
    """
    input_length = int(input_length)
    leave = input_length % n_window
    feat_length = max(0, (leave - 1) // 2 + 1) if leave > 0 else 0
    return (
        max(0, (max(0, (feat_length - 1) // 2 + 1) - 1) // 2 + 1)
        + (input_length // n_window) * 13
    )


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
        n_window: int. Mel-frame chunk half-width used by windowed
            audio encoder. Each chunk spans ``n_window * 2`` mel frames.
            Defaults to `100`.
        n_window_infer: int. Inference-time window size (in mel frames)
            used to derive the after-CNN attention block size. Defaults to
            `400`.
        conv_chunksize: int. CNN batch-chunk size used to bind
            convolution memory. The Keras encoder currently runs the CNN as
            a single pass. Defaults to `500`.
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
        n_window=100,
        n_window_infer=400,
        conv_chunksize=500,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.output_dim = output_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.max_source_positions = max_source_positions
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.dropout = dropout
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize

        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        chunk_length = n_window * 2
        chunk_length_after_cnn = max(1, -(-chunk_length // 8))
        ratio = max(1, n_window_infer // chunk_length)
        self.window_aftercnn = max(1, chunk_length_after_cnn * ratio)

        # === Convolutional downsampling layers ===
        self.conv2d1 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            dtype=dtype,
            name="conv2d1",
        )
        self.conv2d2 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            dtype=dtype,
            name="conv2d2",
        )
        self.conv2d3 = layers.Conv2D(
            downsample_hidden_size,
            kernel_size=3,
            strides=2,
            padding="valid",
            dtype=dtype,
            name="conv2d3",
        )

        self.conv_out = layers.Dense(
            d_model,
            use_bias=False,
            dtype=dtype,
            name="conv_out",
        )

        # === Transformer encoder layers ===
        self.encoder_transformer_layers = [
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

    def _build_window_attention_mask(self, seq_len):
        """Return a block-diagonal self-attention mask.

        The block size is ``self.window_aftercnn`` (see ``__init__``).
        Returns a boolean tensor of shape ``(1, seq_len, seq_len)`` where
        ``True`` positions are attended to. If ``seq_len`` fits within a
        single window the mask is fully ``True`` and broadcasts cleanly.
        """
        window = int(self.window_aftercnn)
        positions = ops.arange(seq_len, dtype="int32")
        block_idx = positions // window
        same_block = ops.equal(
            ops.expand_dims(block_idx, 1),
            ops.expand_dims(block_idx, 0),
        )
        return ops.expand_dims(same_block, axis=0)

    def _call_with_inputs(
        self, input_features, feature_lengths=None, training=False
    ):
        """Encode mel-spectrogram features into output embeddings.

        Args:
            input_features: Tensor with shape
                `(num_audios, time_steps, num_mel_bins)`, batch-padded
                on the time axis.
            feature_lengths: Optional int tensor ``(num_audios,)`` of
                true mel-frame counts per clip (HF's
                ``feature_attention_mask.sum(-1)``). Defaults to every
                clip spanning the full padded length.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(total_valid_positions, output_dim)` —
            the per-clip valid positions concatenated, matching HF.
        """
        # Early-exit for empty inputs (text-only forward through a
        # multimodal backbone).
        batch_size = ops.shape(input_features)[0]
        time_steps = ops.shape(input_features)[1]
        if batch_size == 0 or time_steps == 0:
            return ops.zeros((0, self.output_dim), dtype=self.compute_dtype)

        input_features = ops.cast(input_features, self.compute_dtype)
        lens = (
            [int(time_steps)] * int(batch_size)
            if feature_lengths is None
            else [
                int(v)
                for v in ops.convert_to_numpy(feature_lengths).reshape(-1)
            ]
        )
        if len(lens) != int(batch_size):
            lens = [int(time_steps)] * int(batch_size)
        outputs = [
            self._encode_single(
                input_features[i : i + 1, :n], training=training
            )
            for i, n in enumerate(lens)
        ]
        return ops.concatenate(outputs, axis=0)

    def _conv_downsample(self, hidden_states, training=False):
        """Run the three stride-2 Conv2D layers with HF-style padding.

        HF convs use `padding=1` (symmetric). Keras "same" pads
        asymmetrically (0, 1) on even-length axes, shifting every
        receptive field — so we pad explicitly and the convs are
        configured `padding="valid"`.
        """
        pad2d = [[0, 0], [1, 1], [1, 1], [0, 0]]
        for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
            hidden_states = ops.pad(hidden_states, pad2d)
            hidden_states = conv(hidden_states, training=training)
            hidden_states = ops.gelu(hidden_states)
        return hidden_states

    def _encode_single(self, input_features, training=False):
        """Encode one `(1, time, mels)` clip to `(valid, output_dim)`."""
        batch_size = 1

        # HF splits the mel axis into chunks of `n_window * 2` frames and
        # runs the conv stack per chunk; each full chunk yields
        # `ceil(chunk_len / 8)` tokens and chunk boundaries reset both
        # the conv receptive field and the sinusoidal positions.
        chunk_len = int(self.n_window) * 2
        time_steps = int(ops.shape(input_features)[1])
        n_chunks = -(-time_steps // chunk_len)
        pad = n_chunks * chunk_len - time_steps
        if pad:
            input_features = ops.concatenate(
                [
                    input_features,
                    ops.zeros(
                        (batch_size, pad, self.num_mel_bins),
                        dtype=input_features.dtype,
                    ),
                ],
                axis=1,
            )

        # (batch, chunks, chunk_len, mels) -> (batch*chunks, mels, len, 1)
        # HF convs over (mel, time) spatial dims — mel must be on the
        # height axis so the kernel's spatial offsets line up.
        hidden_states = ops.reshape(
            input_features, (-1, chunk_len, self.num_mel_bins)
        )
        hidden_states = ops.transpose(hidden_states, [0, 2, 1])
        hidden_states = ops.expand_dims(hidden_states, axis=-1)

        hidden_states = self._conv_downsample(hidden_states, training=training)

        # Flatten spatial dimensions and project. HF permutes
        # (chunks, C, f', t') -> (chunks, t', C, f') before flattening.
        chunk_after_cnn = -(-chunk_len // 8)
        hidden_states = ops.transpose(hidden_states, [0, 2, 3, 1])
        flattened = ops.shape(hidden_states)[2] * ops.shape(hidden_states)[3]
        hidden_states = ops.reshape(
            hidden_states, (-1, chunk_after_cnn, flattened)
        )
        hidden_states = self.conv_out(hidden_states)

        # Scale embeddings
        hidden_states = hidden_states * self.embed_scale

        # Add position embeddings (reset per chunk, matching HF).
        positions = ops.cast(
            self.positional_embedding[:chunk_after_cnn, :],
            hidden_states.dtype,
        )
        hidden_states = hidden_states + positions

        # (batch*chunks, chunk_after_cnn, d) -> (batch, chunks*len, d),
        # then keep only the first `feat_len(L_i)` slots of each chunk
        # (the tail chunk is shorter than `chunk_len`). This is what
        # HF's `get_valid_indices` selects.
        hidden_states = ops.reshape(
            hidden_states,
            (batch_size, n_chunks * chunk_after_cnn, self.d_model),
        )
        chunk_lens = np.minimum(
            chunk_len,
            time_steps - np.arange(n_chunks) * chunk_len,
        )
        keep = np.array(
            [
                _get_feat_extract_output_length(int(l), chunk_len)
                for l in chunk_lens
            ],
            dtype=np.int64,
        )
        keep_mask = (
            np.arange(chunk_after_cnn, dtype=np.int64)[None, :] < keep[:, None]
        )
        keep_idx = ops.convert_to_tensor(
            np.where(keep_mask.reshape(-1))[0].astype(np.int64),
            dtype="int32",
        )
        hidden_states = ops.take(hidden_states, keep_idx, axis=1)

        seq_len = ops.shape(hidden_states)[1]
        # Build block-diagonal attention mask
        attention_mask = self._build_window_attention_mask(seq_len)

        # Apply transformer encoder layers
        for encoder_layer in self.encoder_transformer_layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
            )

        # Post-encoder normalization
        hidden_states = self.ln_post(hidden_states)

        # Output projection
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.proj_activation(hidden_states)
        hidden_states = self.proj2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)

        return hidden_states[0]

    def build(self, input_shape=None):
        mel_after_conv1 = (self.num_mel_bins + 1) // 2
        mel_after_conv2 = (mel_after_conv1 + 1) // 2
        mel_after_conv3 = (mel_after_conv2 + 1) // 2

        self.conv2d1.build((None, None, self.num_mel_bins, 1))
        self.conv2d2.build(
            (None, None, mel_after_conv1, self.downsample_hidden_size)
        )
        self.conv2d3.build(
            (None, None, mel_after_conv2, self.downsample_hidden_size)
        )
        self.conv_out.build(
            (None, None, mel_after_conv3 * self.downsample_hidden_size)
        )

        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(self.max_source_positions, self.d_model),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self.positional_embedding.assign(
            _create_sinusoidal_positions(
                self.max_source_positions, self.d_model
            )
        )

        encoder_input_shape = (None, None, self.d_model)
        for encoder_layer in self.encoder_transformer_layers:
            encoder_layer.build(encoder_input_shape)

        self.ln_post.build(encoder_input_shape)
        self.proj1.build(encoder_input_shape)
        self.proj2.build((None, None, self.d_model))
        self.built = True

    def call(self, inputs, training=False):
        """Process a dict of inputs through the audio encoder.

        Args:
            inputs: dict. A dictionary with `"input_features"` key
                containing a mel-spectrogram tensor with shape
                `(num_audios, time_steps, num_mel_bins)`, and an
                optional `"feature_lengths"` int tensor ``(num_audios,)``
                of true mel-frame counts per clip.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(total_valid_positions, output_dim)`.
        """
        return self._call_with_inputs(
            inputs["input_features"],
            feature_lengths=inputs.get("feature_lengths"),
            training=training,
        )

    def compute_output_spec(self, input_spec, **kwargs):
        """Compute output shape for symbolic tracing."""
        input_features_spec = input_spec["input_features"]
        return keras.KerasTensor(
            shape=(None, self.output_dim),
            dtype=input_features_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mel_bins": self.num_mel_bins,
                "d_model": self.d_model,
                "encoder_layers": self.encoder_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "output_dim": self.output_dim,
                "downsample_hidden_size": self.downsample_hidden_size,
                "max_source_positions": self.max_source_positions,
                "scale_embedding": self.scale_embedding,
                "activation_function": self.activation_function,
                "dropout": self.dropout,
                "n_window": self.n_window,
                "n_window_infer": self.n_window_infer,
                "conv_chunksize": self.conv_chunksize,
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
        self.activation = activation

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
                "activation": self.activation,
                "dropout": self.dropout_layer.rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
