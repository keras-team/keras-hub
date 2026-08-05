import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.modernbert.modern_bert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.modernbert.modern_bert_presets import backbone_presets


@keras_hub_export("keras_hub.models.ModernBertBackbone")
class ModernBertBackbone(Backbone):
    """ModernBERT backbone model.

    ModernBERT features Rotary Positional Embeddings (RoPE), GeGLU activations,
    Layer Normalization, and alternating local and global attention layers.

    The backbone accepts a dictionary of token IDs and padding masks, and
    outputs the raw hidden sequence representations.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the GeGLU MLP.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        local_attention_window: int. Window size for local attention layers.
            Defaults to `128`.
        global_attn_every_n_layers: int. Frequency of global attention layers.
            Defaults to `3`.
        dropout: float. Dropout probability for the transformer layers.
            Defaults to `0.0`.
        rotary_max_wavelength: int. Max wavelength for RoPE.
            Defaults to `160000`.
        layer_norm_epsilon: float. Epsilon used by the Layer
        Normalization layers.Defaults to `1e-5`.
        dtype: string or `keras.DTypePolicy`. The dtype of the layers.
            Defaults to `None`.

    Examples:
    ```python
    import keras_hub
    import numpy as np

    # Instantiate a ModernBERT backbone
    backbone = keras_hub.models.ModernBertBackbone(
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    )

    # Prepare dummy input data
    input_data = {
        "token_ids": np.random.randint(0, 50368, size=(2, 512), dtype="int32"),
        "padding_mask": np.ones((2, 512), dtype="int32"),
    }

    # Extract hidden states
    outputs = backbone(input_data)
    ```
    """

    presets = backbone_presets

    def __init__(
        self,
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
        local_attention_window=128,
        global_attn_every_n_layers=3,
        dropout=0.0,
        rotary_max_wavelength=160000,
        local_rotary_max_wavelength=10000,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.local_attention_window = local_attention_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.dropout = dropout
        self.rotary_max_wavelength = rotary_max_wavelength
        self.local_rotary_max_wavelength = local_rotary_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon

        # Dtype handling
        if isinstance(dtype, dict):
            dtype = keras.saving.deserialize_keras_object(dtype)

        if dtype is None:
            layer_dtype_policy = None
        elif isinstance(dtype, keras.DTypePolicy):
            layer_dtype_policy = dtype
        else:
            layer_dtype_policy = keras.DTypePolicy(dtype)

        # Token embedding
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=0.02
            ),
            dtype=dtype,
            name="token_embedding",
        )

        # Embedding normalization
        self.embedding_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            center=False,
            scale=True,
            dtype=dtype,
            name="embedding_norm",
        )

        # Global-attention RoPE
        self.global_rotary_embedding = RotaryEmbedding(
            max_wavelength=rotary_max_wavelength,
            dtype=dtype,
            name="global_rotary_embedding",
        )

        # Local-attention RoPE
        if local_rotary_max_wavelength is None:
            local_rotary_max_wavelength = rotary_max_wavelength

        self.local_rotary_embedding = RotaryEmbedding(
            max_wavelength=local_rotary_max_wavelength,
            dtype=dtype,
            name="local_rotary_embedding",
        )

        self.rotary_embedding = self.global_rotary_embedding

        # Transformer layers
        self.transformer_layers = []

        for i in range(num_layers):
            # ModernBERT uses global attention every Nth layer.
            is_global = (i % global_attn_every_n_layers) == 0

            if is_global:
                attn_window = None
                rotary_embedding = self.global_rotary_embedding
            else:
                attn_window = local_attention_window
                rotary_embedding = self.local_rotary_embedding

            layer = ModernBertEncoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                layer_idx=i,
                rotary_embedding=rotary_embedding,
                local_attention_window=attn_window,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=layer_dtype_policy,
                name=f"transformer_layer_{i}",
            )

            self.transformer_layers.append(layer)

        # Final normalization
        self.final_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layer_norm_epsilon,
            center=False,
            scale=True,
            dtype=dtype,
            name="final_norm",
        )

        # Functional inputs
        token_ids = keras.Input(
            shape=(None,),
            dtype="int32",
            name="token_ids",
        )

        padding_mask = keras.Input(
            shape=(None,),
            dtype="int32",
            name="padding_mask",
        )

        # Embeddings
        x = self.token_embedding(token_ids)
        x = self.embedding_norm(x)

        # Transformer
        for layer in self.transformer_layers:
            x = layer(
                x,
                padding_mask=padding_mask,
            )
        x = self.final_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

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
                "local_rotary_max_wavelength": self.local_rotary_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dtype": keras.saving.serialize_keras_object(self.dtype_policy),
            }
        )

        return config
