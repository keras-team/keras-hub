import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mixtral.mixtral_decoder import (
    MixtralTransformerDecoder,
)
from keras_hub.src.models.mixtral.mixtral_layer_norm import (
    MixtralLayerNormalization,
)


def _mixtral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.MixtralBackbone")
class MixtralBackbone(Backbone):
    """The Mixtral Transformer core architecture with hyperparameters.

    This network implements a mixture of Experts based decoder network,
    Mixtral, as described in
    ["Mixtral of Experts"](https://arxiv.org/pdf/2401.04088).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    Mixtral model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads for
            each transformer.
        hidden_dim (int): The size of the transformer encoding and pooling
            layers.
        intermediate_dim (int): The output dimension of the first Dense layer
            in a three-layer feedforward network for each transformer.
        num_key_value_heads (int): The number of key and value attention heads
            for each transformer.
        rope_max_wavelength (int, optional): The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_scaling_factor (float, optional): The scaling factor for
            calculation of roatary embedding. Defaults to `1.0`.
        layer_norm_epsilon (float, optional): Epsilon for the layer
            normalization layers in the transformer decoder. Defaults to `1e-6`.
        sliding_window (int, optional): The sliding window for the mixtral
            attention layers. This controls the maximum cache size for the
            attention layers in each transformer decoder. Only `sliding_window`
            number of tokens are saved in the cache and used to generate the
            next token. Defaults to `512`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:

    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Mixtral decoder.
    model = keras_hub.models.MixtralBackbone.from_preset("mixtral7b_base_en")
    model(input_data)

    # Randomly initialized Mixtral decoder with custom config.
    model = keras_hub.models.MixtralBackbone(
        vocabulary_size=10,
        hidden_dim=512,
        num_layers=2,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=1024,
        sliding_window=512,
        layer_norm_epsilon=1e-6,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        hidden_dim,
        intermediate_dim,
        num_key_value_heads,
        num_experts,
        top_k=2,
        router_jitter_noise=0.0,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        router_aux_loss_coef=0.02,
        sliding_window=512,
        dropout=0,
        dtype=None,
        output_router_logits=False,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_mixtral_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = MixtralTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                num_experts=num_experts,
                top_k=top_k,
                router_jitter_noise=router_jitter_noise,
                output_router_logits=output_router_logits,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                router_aux_loss_coef=router_aux_loss_coef,
                kernel_initializer=_mixtral_kernel_initializer(stddev=0.02),
                sliding_window=sliding_window,
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = MixtralLayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
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
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
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
        self.num_query_heads = num_query_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        self.rope_max_wavelength = rope_max_wavelength
        self.router_aux_loss_coef = router_aux_loss_coef
        self.rope_scaling_factor = rope_scaling_factor
        self.sliding_window = sliding_window
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "router_jitter_noise": self.router_jitter_noise,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "router_aux_loss_coef": self.router_aux_loss_coef,
                "sliding_window": self.sliding_window,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config
