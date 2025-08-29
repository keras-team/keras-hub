import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.stablelm.stablelm_decoder import (
    StableLMTransformerDecoder,
)


def _stablelm_kernel_initializer(stddev=0.02):
    """Initializer for StableLM kernel weights."""
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.StableLMBackbone")
class StableLMBackbone(Backbone):
    """The StableLM Transformer core architecture with hyperparameters.

    This network implements a Transformer-based decoder network for
    StableLM-3B4E1T, as described in the official documentation. It is a
    decoder-only transformer similar to LLaMA with modifications including
    partial rotary position embeddings and LayerNorm with learned bias terms.
    It includes the embedding lookups and transformer layers.

    The default constructor provides a fully customizable, randomly initialized
    StableLM model with any number of layers, heads, and embedding dimensions.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_query_heads: int. The number of query attention heads.
        hidden_dim: int. The hidden size .
        intermediate_dim: int. The output dimension of the first Dense layer
            in the feedforward network.
        num_key_value_heads: int. The number of key/value attention heads.
        rope_max_wavelength: int. The maximum wavelength for RoPE. Defaults
            to 10000.
        rope_scaling_factor: float. The scaling factor for RoPE. Defaults
            to 1.0.
        layer_norm_epsilon: float. Epsilon for LayerNorm. Defaults to 1e-5.
        dropout: float. Dropout rate. Defaults to 0.0.
        tie_word_embeddings: bool, optional. Whether to tie input and output
            embeddings. Defaults to False.
        dtype: The dtype to use for computations and weights.

    Examples:

    ```python
    # Load a pretrained StableLM backbone.
    model = keras_hub.models.StableLMBackbone.from_preset("stablelm_3b_4e1t_en")

    # Example input data
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Forward pass
    output = model(input_data)
    print(output.shape)  # Expected: (1, 12, 2560)

    # Randomly initialized StableLM decoder with custom config
    model = StableLMBackbone(
        vocabulary_size=50257,
        num_layers=32,
        num_query_heads=32,
        hidden_dim=2560,
        intermediate_dim=6912,
        num_key_value_heads=32,
        rotary_percentage=0.25,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-5,
        dropout=0.0,
        dtype="float32",
    )

    # Example input data
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Forward pass
    output = model(input_data)
    print(output.shape)  # Expected: (1, 12, 2560)
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
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        rotary_percentage=0.25,
        layer_norm_epsilon=1e-5,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_stablelm_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = StableLMTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rotary_percentage=rotary_percentage,
                activation="silu",  # Common activation for modern transformers
                layer_norm_epsilon=layer_norm_epsilon,
                kernel_initializer=_stablelm_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = keras.layers.LayerNormalization(
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
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rotary_percentage = rotary_percentage
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    def get_config(self):
        """Returns the configuration of the model for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rotary_percentage": self.rotary_percentage,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
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

        The returned `LayoutMap` contains the sharding spec for the Llama
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
        layout_map = StableLMBackbone.get_layout_map(
            mesh,
            model_parallel_dim_name="model",
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name='batch',
        )

        with distribution.scope():
           stablelm_model = keras_hub.models.StableLMCausalLM.from_preset()
        ```

        To see how the layout map was applied, load the model then run
        (for one decoder block):
        ```
        embedding_layer = stablelm_model.backbone.get_layer("token_embedding")
        decoder_block_1 = stablelm_model.backbone.get_layer('transformer_layer_0
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
