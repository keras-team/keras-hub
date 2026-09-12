import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer


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


class Qwen3AudioEncoderTransformerLayer(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="gelu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = kernel_initializer

        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.out_proj = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="out_proj",
        )

        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        self.supports_masking = True

    def build(self, inputs_shape=None):
        inputs_shape = (None, None, self.hidden_dim)
        self.q_proj.build(inputs_shape)
        self.k_proj.build(inputs_shape)
        self.v_proj.build(inputs_shape)
        self.out_proj.build(inputs_shape)
        self._self_attention_layer_norm.build(inputs_shape)
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense.build(
            (None, None, self.intermediate_dim)
        )
        self.built = True

    def call(self, inputs, mask=None):
        x = inputs
        residual = x
        x = self._self_attention_layer_norm(x)

        # MHA
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        batch_size = ops.shape(query)[0]
        seq_len = ops.shape(query)[1]

        query = ops.reshape(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        query = ops.transpose(query, (0, 2, 1, 3))  # (B, H, L, D)
        key = ops.reshape(
            key, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.reshape(
            value, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        value = ops.transpose(value, (0, 2, 1, 3))

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_weights = (
            ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * scale
        )

        if mask is not None:
            # mask shape: (B, L)
            mask = ops.cast(mask, query.dtype)
            # Reshape mask for attention: (B, 1, 1, L)
            attention_mask = ops.reshape(mask, (batch_size, 1, 1, seq_len))
            attn_weights = attn_weights + (1.0 - attention_mask) * -1e9

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.hidden_dim)
        )

        x = self.out_proj(attn_output)
        x = x + residual

        residual = x
        x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = x + residual
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


@keras_hub_export("keras_hub.models.Qwen3ASRAudioEncoder")
class Qwen3ASRAudioEncoder(keras.layers.Layer):
    """Qwen3 ASR Audio Encoder."""

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

        self.conv_out = keras.layers.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype_policy,
            name="conv_out",
        )

        self.transformer_layers = []
        for i in range(self.num_layers):
            self.transformer_layers.append(
                Qwen3AudioEncoderTransformerLayer(
                    hidden_dim=self.d_model,
                    intermediate_dim=self.intermediate_dim,
                    num_heads=self.num_attention_heads,
                    dropout=self.dropout,
                    activation="gelu",
                    dtype=self.dtype_policy,
                    name=f"transformer_layer_{i}",
                )
            )

        self.ln_post = keras.layers.LayerNormalization(
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="ln_post",
        )
        self.supports_masking = True

    def build(self, input_features_shape=None, input_features_mask_shape=None):
        freq_bins = self.num_mel_bins
        # Exact PyTorch padding: (I + 2*1 - 3)//2 + 1
        for _ in range(3):
            freq_bins = (freq_bins + 2 - 3) // 2 + 1
        conv_out_dim = self.downsample_hidden_size * freq_bins

        # Positional embedding as constant for exact parameter parity
        self.pos_emb_const = ops.convert_to_tensor(
            compute_sinusoidal_positional_embedding(
                self.max_position_embeddings, self.d_model
            ),
            dtype="float32",
        )

        self.conv2d1.build((None, None, None, 1))
        self.conv2d2.build((None, None, None, self.downsample_hidden_size))
        self.conv2d3.build((None, None, None, self.downsample_hidden_size))
        self.conv_out.build((None, None, conv_out_dim))
        for layer in self.transformer_layers:
            layer.build((None, None, self.d_model))
        self.ln_post.build((None, None, self.d_model))
        self.built = True

    def _post_cnn_length(self, lengths):
        for _ in range(3):
            lengths = ops.where(
                lengths > 0,
                (lengths + 2 - 3) // 2 + 1,
                ops.zeros_like(lengths),
            )
        return lengths

    def call(self, input_features, input_features_mask):
        # input_features: (B, T, F)
        x = input_features
        batch_size = ops.shape(x)[0]
        T = ops.shape(x)[1]

        pad_len = (self.chunk_len - (T % self.chunk_len)) % self.chunk_len
        x = ops.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        input_features_mask = ops.pad(
            input_features_mask, [[0, 0], [0, pad_len]]
        )

        total_T = ops.shape(x)[1]
        num_chunks = total_T // self.chunk_len

        # chunked: (B, num_chunks, chunk_len, F)
        chunked = ops.reshape(
            x, (batch_size, -1, self.chunk_len, self.num_mel_bins)
        )
        # Transpose to (B, num_chunks, F, chunk_len) for Conv2D (N, H, W, C)
        # where H=F, W=chunk_len, C=1
        chunked = ops.transpose(chunked, (0, 1, 3, 2))
        chunked = ops.reshape(
            chunked, (-1, self.num_mel_bins, self.chunk_len, 1)
        )

        # Strictly symmetric pad(1, 1) to match PyTorch Conv2d(..., padding=1)
        conv_out = ops.pad(chunked, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d1(conv_out)
        conv_out = ops.pad(conv_out, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d2(conv_out)
        conv_out = ops.pad(conv_out, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv_out = self.conv2d3(conv_out)

        freq_bins = conv_out.shape[1]
        time_steps = conv_out.shape[2]
        conv_channels = conv_out.shape[3]

        # Permute to (B*num_chunks, T_after_cnn, C, F) ->
        # (B*num_chunks, T_after_cnn, C*F)
        conv_out = ops.transpose(conv_out, (0, 2, 3, 1))
        conv_out = ops.reshape(
            conv_out,
            (-1, time_steps, conv_channels * freq_bins),
        )
        conv_out = self.conv_out(conv_out)

        pos_emb = ops.cast(self.pos_emb_const, self.compute_dtype)
        # Add positional embedding to each chunk (broadcast over B*num_chunks)
        conv_out = conv_out + ops.reshape(
            pos_emb[:time_steps, :], (1, time_steps, self.d_model)
        )

        # Handle mask
        chunk_masks = ops.reshape(
            input_features_mask, (batch_size * num_chunks, self.chunk_len)
        )
        chunk_lens = ops.sum(chunk_masks, axis=-1)
        chunk_aftercnn_lens = self._post_cnn_length(chunk_lens)

        # Windowed attention: Group chunks into windows of size n_window_infer
        # n_window_ratio = 8 if n_window_infer=800, n_window=50
        n_window_ratio = self.n_window_infer // (self.n_window * 2)
        num_windows = (num_chunks + n_window_ratio - 1) // n_window_ratio
        window_size = time_steps * n_window_ratio

        # Pad conv_out to be a multiple of window_size
        total_tokens = num_chunks * time_steps
        pad_tokens = (num_windows * window_size) - total_tokens

        hidden_states = ops.reshape(conv_out, (batch_size, -1, self.d_model))

        # Symbolic-friendly padding
        hidden_states = ops.pad(
            hidden_states, [[0, 0], [0, pad_tokens], [0, 0]]
        )

        # Reshape for windowed processing
        hidden_states = ops.reshape(
            hidden_states, (-1, window_size, self.d_model)
        )

        # Create token-level mask for windowed attention
        # token_mask should have shape (batch_size * num_windows, window_size)
        flat_chunk_aftercnn_lens = ops.reshape(
            chunk_aftercnn_lens, (batch_size, -1)
        )
        flat_chunk_aftercnn_lens = ops.pad(
            flat_chunk_aftercnn_lens,
            [[0, 0], [0, num_windows * n_window_ratio - num_chunks]],
        )

        flat_chunk_aftercnn_lens = ops.reshape(
            flat_chunk_aftercnn_lens, (-1, n_window_ratio)
        )
        # Now we have n_window_ratio chunks per window. Each chunk has
        # 'time_steps' tokens.
        # We need to expand this to (batch_size * num_windows, window_size)

        # For each window, the mask is:
        # [ [1]*L1, [0]*(13-L1), [1]*L2, [0]*(13-L2), ... ]
        token_idx = ops.arange(time_steps, dtype="int32")
        token_idx = token_idx[None, None, :]

        chunk_lens_expanded = ops.expand_dims(flat_chunk_aftercnn_lens, axis=-1)
        window_token_mask = ops.less(token_idx, chunk_lens_expanded)
        window_token_mask = ops.reshape(window_token_mask, (-1, window_size))

        # Windowed processing
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, mask=window_token_mask)

        # Reshape back and truncate padding
        hidden_states = ops.reshape(
            hidden_states, (batch_size, -1, self.d_model)
        )
        hidden_states = hidden_states[:, :total_tokens, :]

        hidden_states = self.ln_post(hidden_states)

        return hidden_states

    def compute_output_spec(self, input_features, input_features_mask):
        T = input_features.shape[1]
        if T is not None:
            num_chunks = (T + self.chunk_len - 1) // self.chunk_len
            output_len = num_chunks * self.max_position_embeddings
        else:
            output_len = None
        return keras.KerasTensor(
            shape=(input_features.shape[0], output_len, self.d_model),
            dtype=self.dtype_policy.compute_dtype,
        )

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


@keras_hub_export("keras_hub.models.Qwen3ASRMultiModalProjector")
class Qwen3ASRMultiModalProjector(keras.layers.Layer):
    def __init__(self, output_dim, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

        self.linear_1 = None
        self.act = keras.layers.Activation(
            self.activation, dtype=self.dtype_policy, name="act"
        )
        self.linear_2 = keras.layers.Dense(
            self.output_dim, dtype=self.dtype_policy, name="linear_2"
        )
        self.supports_masking = True

    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            return
        d_model = input_shape[-1]
        self.linear_1 = keras.layers.Dense(
            d_model, dtype=self.dtype_policy, name="linear_1"
        )
        self.linear_1.build(input_shape)
        # We must call build on linear_2 again because input shape might
        # have changed
        # due to linear_1 and activation.
        self.linear_2.build((None, None, d_model))
        self.built = True

    def call(self, audio_features):
        return self.linear_2(self.act(self.linear_1(audio_features)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"output_dim": self.output_dim, "activation": self.activation}
        )
        return config
