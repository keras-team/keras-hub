import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBERTEncoderLayer,
)
from keras_hub.src.utils.keras_utils import gelu_approximate


@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
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
            dtype=dtype,
            name="rotary_embedding",
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            rms_scaling=True,
            name="embeddings_layer_norm",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = ModernBERTEncoderLayer(
                hidden_size=hidden_dim,
                intermediate_size=intermediate_dim,
                num_heads=num_heads,
                activation=gelu_approximate,
                layer_norm_epsilon=layer_norm_epsilon,
                rotary_embedding=self.position_embedding,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.final_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        x = self.embeddings_layer_norm(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        sequence_output = self.final_norm(x)

        # Instantiate using Functional API Model constructor
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
