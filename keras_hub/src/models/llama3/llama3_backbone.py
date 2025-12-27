from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone


# Config class for Llama3Backbone
@keras_hub_export("keras_hub.models.Llama3BackboneConfig")
class Llama3BackboneConfig:
    """Configuration for Llama3Backbone.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        num_layers: int. Number of transformer layers.
        num_query_heads: int. Number of query attention heads.
        hidden_dim: int. Size of the transformer encoding layers.
        intermediate_dim: int. Output dimension of feedforward network.
        num_key_value_heads: int. Number of key/value attention heads.
        rope_max_wavelength: int. Maximum angular wavelength for RoPE.
        rope_scaling_factor: float. Scaling factor for RoPE.
        layer_norm_epsilon: float. Epsilon for layer normalization.
        dtype: str or DTypePolicy. Dtype for computations and weights.
    """

    def __init__(
        self,
        vocabulary_size=128256,
        num_layers=32,
        num_query_heads=32,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_key_value_heads=8,
        rope_max_wavelength=500000,
        rope_scaling_factor=8.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_key_value_heads = num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dtype = dtype
        # Store any extra kwargs
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_config(self):
        config = {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_query_heads": self.num_query_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_key_value_heads": self.num_key_value_heads,
            "rope_max_wavelength": self.rope_max_wavelength,
            "rope_scaling_factor": self.rope_scaling_factor,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "dtype": self.dtype,
        }
        config.update(self._kwargs)
        return config


# LLaMA 3 shares the same architecture as its predecessors
# So, we simply create an alias for API consistency
@keras_hub_export("keras_hub.models.Llama3Backbone")
class Llama3Backbone(LlamaBackbone):
    """
    The Llama Transformer core architecture with hyperparameters.

    This network implements a Transformer-based decoder network,
    Llama, as described in
    ["Llama 7B"](https://arxiv.org/pdf/2310.06825.pdf).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    Llama model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads for
            each transformer.
        hidden_dim (int): The size of the transformer encoding and pooling
            layers.
        intermediate_dim (int): The output dimension of the first Dense layer in
            a three-layer feedforward network for each transformer.
        num_key_value_heads (int): The number of key and value attention heads
            fo each transformer.
        rope_max_wavelength (int, optional): The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_position_scaling_factor (float, optional): The scaling factor for
            calculation of roatary embedding. Defaults to `1.0`
        rope_requency_adjustment_factor (float, optional): The scaling factor
            used to scale the inverse frequencies.
        rope_low_freq_factor (float, optional): The low frequency factor.
            Defaults to None.
        rope_high_freq_factor: (float, optional) Used for Llama3.1+. The high
            frequency factor. Defaults to None.
        rope_pretraining_sequence_length: (int, optional) Sequence length during
            original pretraining. Defaults to None.
        layer_norm_epsilon (float, optional): Epsilon for the layer
            normalization layers in the transformer decoder. Defaults to `1e-6`.
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

    # Pretrained Llama decoder.
    model = keras_hub.models.Llama3Backbone.from_preset("llama3_8b_en")
    model(input_data)

    # Randomly initialized Llama decoder with custom config.
    model = keras_hub.models.Llama3Backbone(
        vocabulary_size=10,
        hidden_dim=512,
        num_layers=2,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=1024,
        layer_norm_epsilon=1e-6,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    pass
