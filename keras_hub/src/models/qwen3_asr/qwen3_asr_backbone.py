import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
from keras_hub.src.models.qwen3.qwen3_layernorm import Qwen3LayerNorm


def _qwen3_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


class Qwen3ASRInterleaveEmbeddings(keras.layers.Layer):
    """Scatter audio token embeddings into the text embedding sequence."""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, audio_embeddings, text_embeddings, audio_indices):
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        flat_text = ops.reshape(text_embeddings, (-1, self.hidden_dim))
        flat_audio = ops.reshape(audio_embeddings, (-1, self.hidden_dim))

        offsets = ops.arange(batch_size, dtype="int32") * seq_len
        offsets = ops.expand_dims(offsets, axis=-1)

        audio_indices = ops.cast(audio_indices, "int32")
        audio_indices = audio_indices + offsets
        flat_indices = ops.reshape(audio_indices, (-1, 1))

        flat_out = ops.scatter_update(flat_text, flat_indices, flat_audio)
        return ops.reshape(flat_out, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self, audio_embeddings, text_embeddings, audio_indices
    ):
        return keras.KerasTensor(
            shape=text_embeddings.shape,
            dtype=text_embeddings.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras_hub_export("keras_hub.models.Qwen3ASRBackbone")
class Qwen3ASRBackbone(Backbone):
    """The Qwen3-ASR Transformer core architecture with hyperparameters.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads for
            each transformer.
        hidden_dim (int): The size of the transformer encoding and pooling
            layers.
        intermediate_dim (int): The output dimension of the first Dense layer in
            a three-layer feedforward network for each transformer.
        num_key_value_heads (int): The number of key and value attention heads
            for each transformer.
        head_dim (int): The size of each attention head.
        rope_max_wavelength (int, optional): The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `1000000`.
        rope_scaling_factor (float, optional): The scaling factor for
            calculation of rotary embedding. Defaults to `1.0`.
        layer_norm_epsilon (float, optional): Epsilon for the layer
            normalization layers in the transformer decoder. Defaults to `1e-6`.
        dropout (float, optional): Dropout rate for attention and hidden layers.
            Defaults to `0`.
        tie_word_embeddings (bool, optional): Whether to tie input and output
            embeddings. Defaults to `True`.
        sliding_window_size (int, optional): Size of the sliding window for
            attention when enabled. Defaults to `32768`.
        audio_encoder (keras_hub.models.Qwen3ASRAudioEncoder, optional): The
            audio encoder. Defaults to `None`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        vocabulary_size=151936,
        num_layers=28,
        num_query_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_dim=2048,
        intermediate_dim=6144,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        tie_word_embeddings=True,
        sliding_window_size=32768,
        audio_encoder=None,
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

        self.audio_encoder = audio_encoder

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }

        x = self.token_embedding(token_id_input)

        if audio_encoder is not None:
            audio_mel_input = keras.Input(
                shape=(None, audio_encoder.num_mel_bins),
                name="audio_mel",
            )
            audio_mel_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_mel_mask"
            )
            audio_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_indices"
            )

            inputs["audio_mel"] = audio_mel_input
            inputs["audio_mel_mask"] = audio_mel_mask_input
            inputs["audio_indices"] = audio_indices_input

            audio_features = self.audio_encoder(
                audio_mel_input, audio_mel_mask=audio_mel_mask_input
            )

            self.interleave_layer = Qwen3ASRInterleaveEmbeddings(
                hidden_dim=hidden_dim, dtype=dtype, name="interleave_embeddings"
            )
            x = self.interleave_layer(
                audio_embeddings=audio_features,
                text_embeddings=x,
                audio_indices=audio_indices_input,
            )

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
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
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "sliding_window_size": self.sliding_window_size,
                "audio_encoder": None
                if self.audio_encoder is None
                else keras.layers.serialize(self.audio_encoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("audio_encoder") is not None:
            config["audio_encoder"] = keras.layers.deserialize(
                config["audio_encoder"]
            )
        return super().from_config(config)
