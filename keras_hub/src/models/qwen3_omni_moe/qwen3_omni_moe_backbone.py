import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_decoder import (
    Qwen3OmniMoeTransformerDecoder,
)
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_layernorm import Qwen3OmniMoeLayerNorm


def _qwen3_omni_moe_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen3OmniMoeBackbone")
class Qwen3OmniMoeBackbone(Backbone):
    """Qwen3-Omni MoE core network with multimodal capabilities.

    This backbone implements the base Transformer network for the Qwen3-Omni MoE
    model. It includes embedding lookups and transformer layers with a Mixture
    of Experts (MoE) architecture, supporting text, audio, and vision inputs.
    This backbone outputs the final hidden states for each token, not generative
    predictions over the vocabulary space. For higher-level object for text
    generation, see `keras_hub.models.Qwen3OmniMoeCausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Qwen3-Omni MoE model with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of heads for the query projections in
            the attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the attention layer.
        hidden_dim: int. The size of the transformer hidden state at the end of
            each transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the feedforward network for each transformer.
        num_experts: int. The number of experts in each MoE layer.
        num_experts_per_tok: int. The number of experts to select for each token
            in the MoE layer.
        head_dim: int. The size of each attention head.
        layer_norm_epsilon: float. The epsilon value used for every layer norm
            in the transformer model.
        dropout: float. Dropout probability for the transformer encoder.
        sliding_window_size: int. Size of the sliding local window. Defaults to
            4096.
        max_sequence_length: int. The maximum sequence length supported by the
            model. Defaults to 4096.
        dtype: str or `keras.mixed_precision.DTypePolicy`. The dtype to use for
            the model's computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Qwen3-Omni MoE decoder.
    model = keras_hub.models.Qwen3OmniMoeBackbone.from_preset("qwen3_omni_moe_7b")
    model(input_data)

    # Randomly initialized Qwen3-Omni MoE decoder with custom config.
    model = keras_hub.models.Qwen3OmniMoeBackbone(
        vocabulary_size=151936,
        num_layers=32,
        num_query_heads=32,
        num_key_value_heads=4,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_tok=2,
        head_dim=128,
        max_sequence_length=32768,
    )
    model(input_data)
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        head_dim=None,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        sliding_window_size=4096,
        max_sequence_length=32768,
        dtype=None,
        **kwargs,
    ):
        # Set up the config
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_query_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.max_sequence_length = max_sequence_length

        # Token embeddings
        self.token_embedding = ReversibleEmbedding(
            vocabulary_size,
            hidden_dim,
            embeddings_initializer=_qwen3_omni_moe_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )

        # Transformer decoder
        self.transformer_decoder = Qwen3OmniMoeTransformerDecoder(
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            head_dim=head_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            sliding_window_size=sliding_window_size,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
            name="transformer_decoder",
        )

        # Final layer norm
        self.layer_norm = Qwen3OmniMoeLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="layer_norm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        
        # Compute attention mask
        attention_mask = ops.cast(padding_mask_input, dtype="bool")
        
        # Transformer decoder
        decoder_outputs = self.transformer_decoder(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=None,
            cache=None,
            cache_update_index=None,
            training=None,
        )
        
        sequence_output = self.layer_norm(decoder_outputs["hidden_states"])
        
        super().__init__(
            inputs=[token_id_input, padding_mask_input],
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

    def call(
        self,
        inputs,
        position_ids=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        # Handle both dictionary and list inputs (for functional model compatibility)
        if isinstance(inputs, dict):
            token_ids = inputs["token_ids"]
            padding_mask = inputs.get("padding_mask")
        else:
            # inputs is a list from functional model: [token_ids, padding_mask]
            token_ids = inputs[0]
            padding_mask = inputs[1]

        # Embed tokens
        hidden_states = self.token_embedding(token_ids)

        # Compute attention mask
        attention_mask = padding_mask
        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, dtype="bool")
        else:
            attention_mask = ops.ones(
                (ops.shape(token_ids)[0], ops.shape(token_ids)[1]),
                dtype="bool",
            )

        # Transformer decoder
        decoder_outputs = self.transformer_decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )

        # Final layer norm
        hidden_states = self.layer_norm(decoder_outputs["hidden_states"])

        if cache_update_index is not None:
            return hidden_states, decoder_outputs["cache"]
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
