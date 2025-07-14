import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.smollm3.smollm3_layers import SmolLM3DecoderLayer
from keras_hub.src.models.smollm3.smollm3_layers import SmolLM3RotaryEmbedding


@keras_hub_export(
    [
        "keras_hub.models.SmolLM3Backbone",
        "keras_hub.models.SmolLMBackbone",
    ]
)
class SmolLM3Backbone(Backbone):
    """
    The SmolLM Transformer core architecture with hyperparameters.

    This network implements a Transformer-based decoder network,
    SmolLM3, as described in the SmolLM3 model architecture.
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    SmolLM3 model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:


    Examples:

    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained SmolLM decoder.
    model = keras_hub.models.SmolLM3Backbone.from_preset("...")
    model(input_data)

    # Randomly initialized SmolLM3 decoder with custom config.
    model = keras_hub.models.SmolLM3Backbone(
        ...
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
        rms_norm_epsilon,
        layer_norm_epsilon,
        max_position_embeddings,
        rope_theta,
        partial_rotary_factor,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
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
                rms_norm_epsilon=rms_norm_epsilon,
            )
            self.transformer_layers.append(layer)

        self.norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="sequence_output_layernorm",
        )

        self.rotary_embedding = SmolLM3RotaryEmbedding(
            hidden_size=hidden_dim,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        position_embeddings = self.rotary_embedding(x)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=#createcausalmask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        sequence_output = self.layer_norm(x)
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
        self.num_layers = num_layers


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config

