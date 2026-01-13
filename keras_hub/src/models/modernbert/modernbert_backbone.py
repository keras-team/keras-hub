import keras
from keras import layers
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.backbone import Backbone

@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
    """ModernBERT backbone model.

    ModernBERT features Rotary Positional Embeddings (RoPE), GeGLU activations,
    RMSNorm, and Alternating Attention (interleaving local and global layers).
    This backbone provides the raw sequence output from the transformer 
    encoder.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the GeGLU MLP.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        local_attention_window: int. Window size for local attention layers.
            Defaults to `128`.
        global_attn_every_n_layers: int. Frequency of global attention layers.
            Every n-th layer will use global attention. Defaults to `3`.
        dropout: float. Dropout probability for the embeddings and 
            transformer layers. Defaults to `0.0`.
        rotary_max_wavelength: int. The maximum wavelength for the rotary
            positional embeddings. Defaults to `160000`.
        layer_norm_epsilon: float. Epsilon for the RMSNorm layers. 
            Defaults to `1e-5`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype of the
            layers and weights.

    Examples:
    ```python
    import numpy as np
    import keras_hub

    # Instantiate the backbone from scratch.
    backbone = keras_hub.models.ModernBertBackbone(
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    )

    # Predict on random input data.
    batch_size = 2
    seq_length = 512
    token_ids = np.random.randint(0, 50368, (batch_size, seq_length))
    padding_mask = np.ones((batch_size, seq_length))
    outputs = backbone({
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    })

    # Use the backbone in a custom Functional model.
    token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    padding_mask_input = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
    sequence_output = backbone({
        "token_ids": token_id_input,
        "padding_mask": padding_mask_input,
    })
    # Add a classification head
    pooled_output = keras.layers.GlobalAveragePooling1D()(sequence_output)
    outputs = keras.layers.Dense(2, activation="softmax")(pooled_output)
    model = keras.Model(
        inputs=[token_id_input, padding_mask_input],
        outputs=outputs,
    )
    ```
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
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        # === Inputs ===
        # Inside ModernBertBackbone.__init__
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask_input = keras.Input(shape=(None,), dtype="int32", name="padding_mask")

        # ModernBERT uses Rotary Positional Embeddings
        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=rotary_max_wavelength,
            name="rotary_embedding",
        )

        # === Layer Definition & Functional Build ===
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="token_embedding",
        )
        
        # ModernBERT applies RMSNorm immediately after embedding
        embeddings_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            name="embeddings_layer_norm",
        )

        x = token_embedding_layer(token_id_input)
        x = embeddings_layer_norm(x)

        for i in range(num_layers):
            # Global and Local Attention
            is_global = (i + 1) % global_attn_every_n_layers == 0
            current_window = None if is_global else local_attention_window

            x = ModernBertEncoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                rotary_embedding=self.rotary_embedding,
                local_attention_window=current_window,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask_input)

        sequence_output = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            rms_scaling=True,
            name="final_norm",
        )(x)

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
        config.update(
            {
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
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    