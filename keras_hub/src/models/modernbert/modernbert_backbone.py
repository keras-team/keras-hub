import keras
from keras import layers
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.backbone import Backbone

from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertEncoderLayer,
)

from keras_hub.src.utils.keras_utils import gelu_approximate

@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
    """ModernBERT backbone model.

    ModernBERT features Rotary Positional Embeddings (RoPE), GeGLU activations,
    RMSNorm, and Alternating Attention (interleaving local and global layers).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the GeGLU MLP.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        local_attention_window: int. Window size for local attention layers.
            Defaults to `128`.
        global_attn_every_n_layers: int. Frequency of global attention layers.
        dropout: float. Dropout probability.
        rotary_max_wavelength: int. Max wavelength for RoPE.
        layer_norm_epsilon: float. Epsilon for RMSNorm.
        dtype: string or `keras.DTypePolicy`. The dtype of the layers.
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        local_attention_window=128,
        global_attn_every_n_layers=3,
        dropout=0.0,
        rotary_max_wavelength=160000,
        dtype=None,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        # === Inputs ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
             shape=(None,), dtype="int32", name="padding_mask"
        )

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
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

        # === Forward pass with dtype cast ===
        x = self.token_embedding(token_id_input)
        x = self.embeddings_layer_norm(x)
        x = keras.layers.Activation(None, dtype=dtype, name="embeddings_cast")(x)

        for i in range(num_layers):
            is_global = (i + 1) % global_attn_every_n_layers == 0
            current_window = None if is_global else local_attention_window

            # Ensure dtype consistency inside encoder layer
            x = ModernBertEncoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                rotary_embedding=self.position_embedding,
                local_attention_window=current_window,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask_input)
            # Force dtype consistency after each encoder layer
            x = keras.layers.Activation(None, dtype=dtype, name=f"transformer_cast_{i}")(x)

        sequence_output = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            dtype=dtype,
            name="final_norm",
        )(x)
        sequence_output = keras.layers.Activation(None, dtype=dtype, name="sequence_output_cast")(sequence_output)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config Storage ===
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.local_attention_window = local_attention_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.dropout = dropout
        self.rotary_max_wavelength = rotary_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocabulary_size": self.vocabulary_size,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "local_attention_window": self.local_attention_window,
            "global_attn_every_n_layers": self.global_attn_every_n_layers,
            "dropout": self.dropout,
            "rotary_max_wavelength": self.rotary_max_wavelength,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)