import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.smollm3.smollm3_layers import SmolLM3DecoderLayer


@keras_hub_export(
    [
        "keras_hub.models.SmolLM3Backbone",
        "keras_hub.models.SmolLMBackbone",
    ]
)
class SmolLM3Backbone(Backbone):
    """SmolLM3 core network with hyperparameters.

    This network implements a Transformer-based decoder network,
    SmolLM3, as described in the SmolLM3 model architecture.
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    SmolLM3 model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the MLP network of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_attention_heads: int. The number of attention heads for each
            transformer layer.
        num_key_value_heads: int. The number of key-value heads for grouped
            query attention in each transformer layer.
        attention_bias: bool. Whether to use bias in the query, key, value, and
            output projection layers in the attention blocks.
        attention_dropout: float. Dropout probability for the attention layers.
        rope_layer_enabled_list: list of bool. List indicating whether RoPE
            (Rotary Position Embedding) is enabled for each layer. Typically,
            some layers may disable RoPE for architectural variations.
        layer_types: list of str. List of layer types for each transformer
            layer (e.g., "attention" or other custom types).
        mlp_bias: bool. Whether to use bias in the MLP (feedforward) layers.
        layer_norm_epsilon: float. Epsilon value for layer normalization layers
            to prevent division by zero.
        max_position_embeddings: int. The maximum sequence length that this
            model might ever be used with.
        rope_theta: float. The base period of the RoPE embeddings.
        partial_rotary_factor: float. The percentage of hidden dimensions to
            rotate in RoPE. A value of 1.0 rotates all dimensions, while values
            less than 1.0 only rotate a subset.

    Examples:

    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained SmolLM3 decoder.
    model = keras_hub.models.SmolLM3Backbone.from_preset(
        "hf://HuggingFaceTB/SmolLM3-3B"
    )
    model(input_data)

    # Randomly initialized SmolLM3 decoder with custom config.
    model = keras_hub.models.SmolLM3Backbone(
        vocabulary_size=49152,
        hidden_dim=576,
        intermediate_dim=1536,
        num_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        attention_bias=False,
        attention_dropout=0.0,
        rope_layer_enabled_list=[True] * 30,
        layer_types=["attention"] * 30,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        attention_bias,
        attention_dropout,
        rope_layer_enabled_list,
        layer_types,
        mlp_bias,
        layer_norm_epsilon,
        max_position_embeddings,
        rope_theta,
        partial_rotary_factor,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = SmolLM3DecoderLayer(
                hidden_size=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                rope_layer_enabled_list=rope_layer_enabled_list,
                layer_types=layer_types,
                layer_idx=i,
                intermediate_size=intermediate_dim,
                mlp_bias=mlp_bias,
                layer_norm_epsilon=layer_norm_epsilon,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="sequence_output_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )

        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)

        for decoder_layer in self.transformer_layers:
            x = decoder_layer(
                x,
                decoder_padding_mask=padding_mask_input,
                **kwargs,
            )

        sequence_output = self.norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_layer_enabled_list = rope_layer_enabled_list
        self.layer_types = layer_types
        self.mlp_bias = mlp_bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "rope_layer_enabled_list": self.rope_layer_enabled_list,
                "layer_types": self.layer_types,
                "mlp_bias": self.mlp_bias,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "max_position_embeddings": self.max_position_embeddings,
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
        )
        return config
