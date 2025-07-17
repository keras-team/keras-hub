from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone


@keras_hub_export("keras_hub.models.Phi4Backbone")
class Phi4Backbone(Phi3Backbone):
    """Phi-4 core network with hyperparameters.

    This network implements a Transformer-based decoder network,
    Phi-4, as described in ["Phi-4 Technical Report"](https://arxiv.org/pdf/2412.08905).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    phi-4 model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Note that the defaults here are the Phi-3 defaults, because the Phi-4 model
    follows the Phi-3-medium architecture but with different hyper-parameters.
    Use `keras_hub.models.Backbone.from_preset` to get the Phi-4 defaults.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The size of the embeddings and the hidden states of
            the transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a three-layer feedforward network for each transformer.
        num_query_heads: int. The number of query attention heads for each
            transformer layer.
        num_key_value_heads: int. The number of key and value attention heads
            for each transformer layer.
        layer_norm_epsilon: float, optional. Epsilon for the RMS layernorm
            layers in the transformer decoder. Defaults to `1e-6`.
        dropout:: float, optional. Dropout probability for the Transformer
            decoder.
        max_sequence_length: int, optional. The maximum sequence length
            that this model might ever be used with. Defaults to `4096`.
        pretraining_sequence_length: int, optional. The maximum sequence length
            that the model was pretrained with. Defaults to `4096`.
        rope_max_wavelength: int, optional. The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_scaling_type: str, optional. The type of the rope scaling. Can be
            either `None` or `"su"`. `None` is for no rope scaling, `"su"` is
            for SuScaled rope, `"su"` is used when `max_sequence_length` is
            larger than `original_max_sequence_length`. Defaults to `None`.
        rope_scaling_short_factor: list[float]. List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is smaller than `pretraining_sequence_length`.
            Defaults to `None`.
        rope_scaling_long_factor: list[float]. List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is larger than `pretraining_sequence_length`.
            Defaults to `None`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.
    """

    pass
