import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
from keras_hub.src.models.qwen3.qwen3_layernorm import Qwen3LayerNorm
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRMultiModalProjector,
)


def _qwen3_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


class Qwen3ASRScatterAudio(keras.layers.Layer):
    def __init__(self, audio_token_id, **kwargs):
        super().__init__(**kwargs)
        self.audio_token_id = audio_token_id

    def get_config(self):
        config = super().get_config()
        config.update({"audio_token_id": self.audio_token_id})
        return config

    def call(self, inputs_embeds, audio_embeds, token_ids):
        batch_size = ops.shape(token_ids)[0]
        seq_len = ops.shape(token_ids)[1]
        hidden_dim = audio_embeds.shape[2]

        special_audio_mask = ops.equal(token_ids, self.audio_token_id)

        # Compute which audio feature belongs to which placeholder
        cum_mask = (
            ops.cumsum(ops.cast(special_audio_mask, "int32"), axis=-1) - 1
        )
        max_index = ops.maximum(ops.shape(audio_embeds)[1] - 1, 0)
        safe_cum_mask = ops.clip(cum_mask, 0, max_index)

        # Expand and broadcast safe_cum_mask to match audio_embeds shape
        safe_cum_mask = ops.expand_dims(safe_cum_mask, axis=-1)
        safe_cum_mask = ops.broadcast_to(
            safe_cum_mask, (batch_size, seq_len, hidden_dim)
        )

        # Gather potential updates along axis 1 (sequence length)
        potential_updates = ops.take_along_axis(
            audio_embeds, safe_cum_mask, axis=1
        )

        # Use ops.where to scatter between potential updates and inputs_embeds
        special_audio_mask_ex = ops.expand_dims(special_audio_mask, axis=-1)
        out = ops.where(special_audio_mask_ex, potential_updates, inputs_embeds)
        return out

    def compute_output_spec(self, inputs_embeds, audio_embeds, token_ids):
        return keras.KerasTensor(
            shape=inputs_embeds.shape,
            dtype=inputs_embeds.dtype,
        )


@keras_hub_export("keras_hub.models.Qwen3ASRBackbone")
class Qwen3ASRBackbone(Backbone):
    """The Qwen3 ASR Transformer backbone.

    This network implements the Qwen3-ASR architecture, which combines a
    Qwen3ASRAudioEncoder and a Qwen3 text decoder.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of text decoder layers.
        num_query_heads: int. The number of query attention heads for the text
        decoder.
        num_key_value_heads: int. The number of key and value attention heads
            for the text decoder.
        head_dim: int. Dimension of each attention head for the text decoder.
        hidden_dim: int. The size of the text decoder hidden dimension.
        intermediate_dim: int. The FFN intermediate dimension for the
        text decoder.
        rope_max_wavelength: int, optional. The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_scaling_factor: float, optional. The scaling factor for
            calculation of rotary embedding. Defaults to `1.0`.
        layer_norm_epsilon: float, optional. Epsilon for the layer
            normalization layers. Defaults to `1e-6`.
        dropout: float, optional. Dropout rate. Defaults to `0.0`.
        tie_word_embeddings: bool, optional. Whether to tie input and output
            embeddings. Defaults to `True`.
        sliding_window_size: int, optional. Size of the sliding window for
            attention when enabled. Defaults to `32768`.

        audio_num_mel_bins: int. Number of mel bins in the input spectrogram.
            Defaults to `128`.
        audio_num_layers: int. Number of audio encoder layers. Defaults to `24`.
        audio_num_attention_heads: int. Number of audio encoder attention heads.
            Defaults to `16`.
        audio_intermediate_dim: int. FFN intermediate dimension for the audio
            encoder. Defaults to `4096`.
        audio_d_model: int. Hidden dimension of the audio encoder.
            Defaults to `1024`.
        audio_n_window: int. Half the chunk size for the audio encoder.
            Defaults to `50`.
        audio_n_window_infer: int. Inference window size for the audio encoder.
            Defaults to `800`.
        audio_downsample_hidden_size: int. Hidden size of conv stem in the
            audio encoder. Defaults to `480`.
        audio_max_position_embeddings: int. Max position embeddings for the
            audio encoder chunk. Defaults to `13`.
        audio_token_id: int. The token ID used as a placeholder for audio
            features. Defaults to `151676`.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers=28,
        num_query_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_dim=1024,
        intermediate_dim=3072,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=True,
        sliding_window_size=32768,
        audio_num_mel_bins=128,
        audio_num_layers=18,
        audio_num_attention_heads=14,
        audio_intermediate_dim=3584,
        audio_d_model=896,
        audio_n_window=50,
        audio_n_window_infer=800,
        audio_downsample_hidden_size=480,
        audio_max_position_embeddings=13,
        audio_token_id=151676,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen3_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )

        self.audio_encoder = Qwen3ASRAudioEncoder(
            num_mel_bins=audio_num_mel_bins,
            num_layers=audio_num_layers,
            num_attention_heads=audio_num_attention_heads,
            intermediate_dim=audio_intermediate_dim,
            d_model=audio_d_model,
            dropout=dropout,
            n_window=audio_n_window,
            n_window_infer=audio_n_window_infer,
            downsample_hidden_size=audio_downsample_hidden_size,
            max_position_embeddings=audio_max_position_embeddings,
            dtype=dtype,
            name="audio_encoder",
        )

        self.projector = Qwen3ASRMultiModalProjector(
            output_dim=hidden_dim,
            dtype=dtype,
            name="projector",
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

        self.scatter_audio = Qwen3ASRScatterAudio(
            audio_token_id,
            dtype=dtype,
            name="scatter_audio",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        audio_mel_input = keras.Input(
            shape=(None, audio_num_mel_bins), name="audio_mel"
        )
        audio_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="audio_mask"
        )

        inputs_embeds = self.token_embedding(token_id_input)

        # Audio encoding
        audio_embeds = self.audio_encoder(audio_mel_input, audio_mask_input)
        audio_embeds = self.projector(
            audio_embeds
        )  # (B, audio_seq_len, hidden_dim)

        x = self.scatter_audio(inputs_embeds, audio_embeds, token_id_input)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "audio_mel": audio_mel_input,
                "audio_mask": audio_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_heads = num_key_value_heads
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.sliding_window_size = sliding_window_size

        self.audio_num_mel_bins = audio_num_mel_bins
        self.audio_num_layers = audio_num_layers
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_intermediate_dim = audio_intermediate_dim
        self.audio_d_model = audio_d_model
        self.audio_n_window = audio_n_window
        self.audio_n_window_infer = audio_n_window_infer
        self.audio_downsample_hidden_size = audio_downsample_hidden_size
        self.audio_max_position_embeddings = audio_max_position_embeddings
        self.audio_token_id = audio_token_id

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "sliding_window_size": self.sliding_window_size,
                "audio_num_mel_bins": self.audio_num_mel_bins,
                "audio_num_layers": self.audio_num_layers,
                "audio_num_attention_heads": self.audio_num_attention_heads,
                "audio_intermediate_dim": self.audio_intermediate_dim,
                "audio_d_model": self.audio_d_model,
                "audio_n_window": self.audio_n_window,
                "audio_n_window_infer": self.audio_n_window_infer,
                "audio_downsample_hidden_size": (
                    self.audio_downsample_hidden_size
                ),
                "audio_max_position_embeddings": (
                    self.audio_max_position_embeddings
                ),
                "audio_token_id": self.audio_token_id,
            }
        )
        return config
