import keras
from keras import layers
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.modernbert.modernbert_layers import ModernBertEncoderLayer
from keras_hub.src.models.backbone import Backbone

@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
    """ModernBERT backbone model.

    ModernBERT is a modernized BERT architecture featuring Rotary Positional 
    Embeddings (RoPE), GeGLU activations, RMSNorm, and Alternating Attention 
    (mixing global and local sliding window attention).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the GeGLU MLP.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        dropout: float. Dropout probability.
        local_attention_window: int. The window size for local attention.
        global_attn_every_n_layers: int. Frequency of global attention layers.
        rotary_max_wavelength: int. The maximum wavelength for RoPE.
        layer_norm_epsilon: float. Epsilon for the RMSNorm layers.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use 
            for model computations and weights.
        **kwargs: Standard `keras_hub.models.Backbone` arguments.

    Examples:
    ```python
    input_data = {
        "token_ids": keras.ops.ones((2, 128), dtype="int32"),
        "padding_mask": keras.ops.ones((2, 128), dtype="int32"),
    }
    model = keras_hub.models.ModernBertBackbone(
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    )
    outputs = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        dropout=0.0,
        local_attention_window=128,
        global_attn_every_n_layers=3,
        rotary_max_wavelength=160000,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        # Layer Definitions
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            dtype=dtype,
            name="token_embedding",
        )

        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=rotary_max_wavelength,
            dtype=dtype,
            name="rotary_embedding",
        )

        # RMSNorm
        self.embeddings_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            dtype=dtype,
            name="embeddings_layer_norm",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            # Alternating Attention Logic:
            # Every n-th layer is Global, others are Local.
            is_global = (i + 1) % global_attn_every_n_layers == 0
            current_window = None if is_global else local_attention_window
            
            self.transformer_layers.append(
                ModernBertEncoderLayer(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_heads=num_heads,
                    rotary_embedding=self.rotary_embedding,
                    dropout=dropout,
                    local_attention_window=current_window,
                    layer_norm_epsilon=layer_norm_epsilon,
                    dtype=dtype,
                    name=f"transformer_layer_{i}",
                )
            )
        self.final_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            dtype=dtype,
            name="final_norm",
        )

        # Functional API Call
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask_input = keras.Input(shape=(None,), dtype="int32", name="padding_mask")

        x = self.token_embedding(token_id_input)
        x = self.embeddings_layer_norm(x)
        
        for layer in self.transformer_layers:
            x = layer(x, padding_mask=padding_mask_input)
        
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

        # Internal Attributes
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.local_attention_window = local_attention_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.rotary_max_wavelength = rotary_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "local_attention_window": self.local_attention_window,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config