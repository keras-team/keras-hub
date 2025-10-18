import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen_moe.qwen_moe_decoder import (
    QwenMoeTransformerDecoder,
)


def _qwen_moe_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export(
    "keras_hub.models.QwenMoeBackbone",
)
class QwenMoeBackbone(Backbone):
    """Qwen MoE core network with hyperparameters.

    This backbone implements the base Transformer network for the Qwen MoE
    model. It includes embedding lookups and transformer layers with a Mixture
    of Experts (MoE) architecture, where each layer uses a sparse set of experts
    for efficient computation. This backbone outputs the final hidden states for
    each token, not generative predictions over the vocabulary space. For higher
    -level object for text generation, see `keras_hub.models.QwenMoeCausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Qwen MoE model with any number of layers, heads, and embedding dimensions.
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
        moe_intermediate_dim: int. The intermediate dimension for each expert
            in the MoE feedforward network.
        shared_expert_intermediate_dim: int. The intermediate dimension for the
            shared expert in the MoE feedforward network.
        num_experts: int. The number of experts in each MoE layer.
        top_k: int. The number of top experts to select for each token in the
            MoE layer.
        head_dim: int. The size of each attention head.
        layer_norm_epsilon: float. The epsilon value used for every layer norm
            in the transformer model.
        dropout: float. Dropout probability for the transformer encoder.
        use_sliding_window_attention: bool. Whether to use sliding local window
            attention. Defaults to False.
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

    # Pretrained Qwen MoE decoder.
    model = keras_hub.models.QwenMoeBackbone.from_preset("qwen_moe_a2_7b")
    model(input_data)

    # Randomly initialized Qwen MoE decoder with custom config.
    model = keras_hub.models.QwenMoeBackbone(
        vocabulary_size=151936,
        num_layers=28,
        num_query_heads=16,
        num_key_value_heads=8,
        hidden_dim=2048,
        intermediate_dim=4096,
        moe_intermediate_dim=128,
        shared_expert_intermediate_dim=4096,
        num_experts=60,
        top_k=4,
        head_dim=128,
        max_sequence_length=4096,
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
        moe_intermediate_dim,
        shared_expert_intermediate_dim,
        num_experts,
        top_k=4,
        norm_top_k_prob=False,
        decoder_sparse_step=1,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,
        tie_word_embeddings=False,
        use_sliding_window_attention=False,
        sliding_window_size=32768,
        output_router_logits=False,
        router_aux_loss_coefficient=0.001,
        mlp_only_layers=[],
        training=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen_moe_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = QwenMoeTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                moe_intermediate_dim=moe_intermediate_dim,
                shared_expert_intermediate_dim=shared_expert_intermediate_dim,
                num_experts=num_experts,
                top_k=top_k,
                norm_top_k_prob=norm_top_k_prob,
                decoder_sparse_step=decoder_sparse_step,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen_moe_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                use_sliding_window_attention=use_sliding_window_attention,
                sliding_window_size=sliding_window_size,
                output_router_logits=output_router_logits,
                router_aux_loss_coefficient=router_aux_loss_coefficient,
                mlp_only_layers=mlp_only_layers,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = QwenLayerNorm(
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
            x = transformer_layer(
                x, decoder_padding_mask=padding_mask_input, training=training
            )
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
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_dim = shared_expert_intermediate_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_heads = num_key_value_heads
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_top_k_prob = norm_top_k_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.output_router_logits = output_router_logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "shared_expert_intermediate_dim": (
                    self.shared_expert_intermediate_dim
                ),
                "rope_max_wavelength": self.rope_max_wavelength,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "norm_top_k_prob": self.norm_top_k_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
                "output_router_logits": self.output_router_logits,
            }
        )
        return config

    @staticmethod
    def get_layout_map(
        device_mesh,
        model_parallel_dim_name="model",
        data_parallel_dim_name="batch",
    ):
        """Get a `keras.distribution.LayoutMap` for model parallel distribution.

        The returned `LayoutMap` contains the sharding spec for the QwenMoe
        backbone weights, so that you can use it to distribute weights across
        the accelerators.

        Example:
        ```
        # Feel free to change the mesh shape to balance data and model
        # parallelism
        mesh = keras.distribution.DeviceMesh(
            shape=(1, 8),
            axis_names=('batch', 'model'),
            devices=keras.distribution.list_devices(),
        )
        layout_map = QwenMoeBackbone.get_layout_map(
            mesh,
            model_parallel_dim_name="model",
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name='batch',
        )

        with distribution.scope():
           qwen_moe_model = keras_hub.models.QwenMoeBackbone.from_preset()
        ```

        To see how the layout map was applied, load the model then run
        (for one decoder block):
        ```
        embedding_layer = qwen_moe_model.backbone.get_layer("token_embedding")
        decoder_block_1 = qwen_moe_model.backbone.get_layer(
            'transformer_layer_0'
        )
        for variable in embedding_layer.weights + decoder_block_1.weights:
            print(
                f'{variable.path:<58}  {str(variable.shape):<16}  '
                f'{str(variable.value.sharding.spec)}'
            )
        ```

        Args:
            device_mesh: The `keras.distribution.DeviceMesh` instance for
                distribution.
            model_parallel_dim_name: The axis name of the device mesh, where
                the weights should be partition on.
            data_parallel_dim_name: The axis name of the device mesh, where
                the data should be partition on.
        Return:
            `keras.distribution.LayoutMap` that contains the sharding spec
            for all the model weights.
        """
        # The weight path and shape of the Llama backbone is like below
        # token_embedding/embeddings                              (128256, 2048)
        # repeat block for decoder
        # transformer_layer_0/self_attention/query/kernel         (2048, 32, 64)
        # transformer_layer_0/self_attention/key/kernel           (2048, 8, 64)
        # transformer_layer_0/self_attention/value/kernel         (2048, 8, 64)
        # transformer_layer_0/self_attention/attention_output/kernel
        #                                                         (32, 64, 2048)
        # transformer_layer_0/self_attention_layernorm/scale      (2048,)
        # transformer_layer_0/feedforward_intermediate_dense/kernel
        #                                                         (2048, 8192)
        # transformer_layer_0/feedforward_gate_dense/kernel       (2048, 8192)
        # transformer_layer_0/feedforward_output_dense/kerne      (8192, 2048)
        # transformer_layer_0/feedforward_layernorm/scale         (2048,)

        if not isinstance(device_mesh, keras.distribution.DeviceMesh):
            raise ValueError(
                "Invalid device_mesh type. Expected "
                f"`keras.distribution.Device`, got {type(device_mesh)}"
            )
        if model_parallel_dim_name not in device_mesh.axis_names:
            raise ValueError(
                f"{model_parallel_dim_name} is not found in the "
                f"device_mesh.axis_names. {device_mesh.axis_name=}"
            )
        if data_parallel_dim_name not in device_mesh.axis_names:
            raise ValueError(
                f"{data_parallel_dim_name} is not found in the "
                f"device_mesh.axis_names. {device_mesh.axis_name=}"
            )
        # Note that it is possible to further config the mesh to be 3D, eg
        # (data, seq, model). We leave it as 2D for now for simplicity.
        data_dim = data_parallel_dim_name
        model_dim = model_parallel_dim_name
        # The sharding config is based on the Gemma team training config.
        # See https://arxiv.org/abs/2403.08295
        layout_map = keras.distribution.LayoutMap(device_mesh)
        layout_map["token_embedding/embeddings"] = (model_dim, data_dim)
        layout_map[
            "transformer_layer.*self_attention.*(query|key|value).kernel"
        ] = (
            model_dim,
            data_dim,
            None,
        )
        layout_map["transformer_layer.*attention_output.kernel"] = (
            model_dim,
            None,
            data_dim,
        )
        layout_map[
            "transformer_layer.*feedforward_intermediate_dense.kernel"
        ] = (
            data_dim,
            model_dim,
        )
        layout_map["transformer_layer.*feedforward_gate_dense.kernel"] = (
            data_dim,
            model_dim,
        )
        layout_map["transformer_layer.*feedforward_output_dense.kernel"] = (
            model_dim,
            data_dim,
        )

        return layout_map
