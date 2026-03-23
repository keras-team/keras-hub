import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
from keras_hub.src.models.qwen3.qwen3_layernorm import Qwen3LayerNorm
from keras_hub.src.models.qwen3_asr.qwen3_asr_encoder import Qwen3ASREncoder


def _qwen3_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras.saving.register_keras_serializable(package="keras_hub")
class ReplaceAudioEmbeddings(keras.layers.Layer):
    """Replace token embeddings at audio placeholder positions.

    The Qwen3-ASR architecture uses replacement embedding: audio encoder
    outputs are scattered into the positions of ``<|AUDIO|>`` placeholder
    tokens in the full token sequence.

    Args:
        audio_token_id: int. Token ID of the audio placeholder.
    """

    def __init__(self, audio_token_id, **kwargs):
        super().__init__(**kwargs)
        self.audio_token_id = audio_token_id

    def call(self, embeddings, audio_embeddings, token_ids):
        audio_mask = ops.equal(token_ids, self.audio_token_id)  # (B, S)

        # Build indices mapping each audio mask position to the
        # corresponding audio embedding index.
        cum = ops.cumsum(ops.cast(audio_mask, "int32"), axis=1)
        indices = ops.clip(cum - 1, 0, ops.shape(audio_embeddings)[1] - 1)

        # Gather audio embeddings aligned to the full sequence.
        indices_3d = ops.broadcast_to(
            ops.expand_dims(indices, -1),
            ops.shape(embeddings),
        )
        audio_aligned = ops.take_along_axis(
            audio_embeddings, indices_3d, axis=1
        )

        # Select audio embeddings where mask is True, text otherwise.
        mask_3d = ops.expand_dims(audio_mask, -1)
        return ops.where(mask_3d, audio_aligned, embeddings)

    def get_config(self):
        config = super().get_config()
        config["audio_token_id"] = self.audio_token_id
        return config


@keras_hub_export("keras_hub.models.Qwen3ASRBackbone")
class Qwen3ASRBackbone(Backbone):
    """Qwen3-ASR core network with hyperparameters.

    This backbone implements the Qwen3-ASR multimodal architecture, which
    combines an Audio Transformer (AuT) encoder with a Qwen3 text decoder
    for automatic speech recognition.

    The forward pass works as follows:

    1. All token IDs (including ``<|AUDIO|>`` placeholders) are embedded.
    2. Mel spectrogram features are processed by the audio encoder,
       producing a sequence of audio embeddings projected to the decoder
       hidden dimension.
    3. Audio embeddings **replace** the embeddings at ``<|AUDIO|>``
       placeholder positions (replacement embedding).
    4. The combined sequence is passed through Qwen3 decoder layers with
       causal attention.
    5. A final RMS normalization produces the output hidden states.

    The default constructor gives a fully customizable, randomly initialized
    model. To load preset architectures and weights, use ``from_preset``.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of Transformer decoder layers.
        num_query_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key/value attention heads
            (for grouped-query attention).
        head_dim: int. The dimension of each attention head.
        hidden_dim: int. The hidden size of the decoder.
        intermediate_dim: int. The intermediate dimension of the gated
            feedforward network in each decoder layer.
        num_mel_bins: int. The number of mel frequency bins in the input
            spectrogram. Defaults to ``128``.
        encoder_d_model: int. The hidden size of the audio encoder.
            Defaults to ``1024``.
        encoder_num_layers: int. The number of audio encoder transformer
            layers. Defaults to ``24``.
        encoder_attention_heads: int. The number of attention heads in
            the audio encoder. Defaults to ``16``.
        encoder_ffn_dim: int. The feedforward intermediate dimension in
            each encoder layer. Defaults to ``4096``.
        downsample_hidden_size: int. The number of Conv2D filters for audio
            downsampling. Defaults to ``480``.
        audio_token_id: int. Token ID of the ``<|AUDIO|>`` placeholder.
            Defaults to ``151676``.
        n_window: int. Half the chunk size in mel frames for the audio
            encoder. Defaults to ``50``.
        max_source_positions: int. Maximum position embedding length for the
            audio encoder. Defaults to ``1500``.
        rope_max_wavelength: int. The maximum angular wavelength for rotary
            position embeddings. Defaults to ``1000000``.
        rope_scaling_factor: float. The scaling factor for rotary position
            embeddings. Defaults to ``1.0``.
        layer_norm_epsilon: float. Epsilon for layer normalization.
            Defaults to ``1e-6``.
        dropout: float. Dropout rate. Defaults to ``0.0``.
        tie_word_embeddings: bool. Whether to tie the input and output
            token embeddings. Defaults to ``True``.
        sliding_window_size: int. The sliding window size for attention.
            Defaults to ``32768``.
        dtype: string or ``keras.mixed_precision.DTypePolicy``. The dtype
            to use for model computations and weights.

    Example:
    ```python
    input_data = {
        "audio_features": np.random.uniform(size=(1, 800, 128)),
        "token_ids": np.array([[151676] * 100 + [1, 2, 3]]),
        "padding_mask": np.array([[1] * 103]),
    }
    model = keras_hub.models.Qwen3ASRBackbone(
        vocabulary_size=151936,
        num_layers=2,
        num_query_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_dim=2048,
        intermediate_dim=6144,
        num_mel_bins=128,
        encoder_d_model=64,
        encoder_num_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        downsample_hidden_size=16,
        dtype="float32",
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        num_mel_bins=128,
        encoder_d_model=1024,
        encoder_num_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        downsample_hidden_size=480,
        audio_token_id=151676,
        n_window=50,
        n_window_infer=800,
        max_source_positions=1500,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=True,
        sliding_window_size=32768,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.audio_encoder = Qwen3ASREncoder(
            num_mel_bins=num_mel_bins,
            d_model=encoder_d_model,
            encoder_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            downsample_hidden_size=downsample_hidden_size,
            output_dim=hidden_dim,
            n_window=n_window,
            n_window_infer=n_window_infer,
            max_source_positions=max_source_positions,
            dtype=dtype,
            name="audio_encoder",
        )
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen3_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.audio_replacer = ReplaceAudioEmbeddings(
            audio_token_id=audio_token_id,
            dtype=dtype,
            name="audio_replacer",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = Qwen3TransformerDecoder(
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen3_kernel_initializer(stddev=0.02),
                dropout=dropout,
                sliding_window_size=sliding_window_size,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = Qwen3LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # === Functional Model ===
        audio_input = keras.Input(
            shape=(None, num_mel_bins), name="audio_features"
        )
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed all tokens (including audio placeholders).
        embeddings = self.token_embedding(token_id_input)

        # Encode audio and replace placeholder embeddings.
        audio_embeddings = self.audio_encoder(audio_input)
        x = self.audio_replacer(embeddings, audio_embeddings, token_id_input)

        # Run through the decoder.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs={
                "audio_features": audio_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_mel_bins = num_mel_bins
        self.encoder_d_model = encoder_d_model
        self.encoder_num_layers = encoder_num_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.audio_token_id = audio_token_id
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.max_source_positions = max_source_positions
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.sliding_window_size = sliding_window_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_mel_bins": self.num_mel_bins,
                "encoder_d_model": self.encoder_d_model,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "downsample_hidden_size": self.downsample_hidden_size,
                "audio_token_id": self.audio_token_id,
                "n_window": self.n_window,
                "n_window_infer": self.n_window_infer,
                "max_source_positions": self.max_source_positions,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
