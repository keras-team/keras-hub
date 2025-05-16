import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.utils.keras_utils import gelu_approximate


@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
    """A ModernBERT encoder network.

    This class implements the ModernBERT backbone, using rotary embeddings,
    RMS normalization, and a stack of TransformerEncoder layers.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        max_sequence_length=8192,
        dropout=0.0,
        rotary_max_wavelength=160000.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = RotaryEmbedding(
            max_wavelength=rotary_max_wavelength,
            sequence_axis=1,
            feature_axis=-1,
            dtype=dtype,
            name="rotary_embedding",
        )
        self.embeddings_layer_norm = RMSNormalization(
            dtype=dtype,
            epsilon=layer_norm_epsilon,
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="embeddings_dropout"
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=gelu_approximate,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=0.02
                ),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.final_norm = RMSNormalization(
            dtype=dtype,
            epsilon=layer_norm_epsilon,
            name="final_normalization",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens and apply rotary position embedding
        x = self.token_embedding(token_id_input)
        x = self.position_embedding(x)
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)

        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)

        # Final normalization
        sequence_output = self.final_norm(x)

        super().__init__(
            inputs={
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
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.rotary_max_wavelength = rotary_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "max_sequence_length": self.max_sequence_length,
                "dropout": self.dropout,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
