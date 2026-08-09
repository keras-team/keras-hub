from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.roberta import roberta_backbone


@keras_hub_export("keras_hub.models.XLMRobertaBackbone")
class XLMRobertaBackbone(roberta_backbone.RobertaBackbone):
    """An XLM-RoBERTa encoder network.

    This class implements a bi-directional Transformer-based encoder as
    described in ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but it does not
    include the masked language modeling head used during pretraining.

    The default constructor gives a fully customizable, randomly initialized
    RoBERTa encoder with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset()`
    constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length` default value. This determines the variable
            shape for positional embeddings.
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

    # Pretrained XLM-R encoder.
    model = keras_hub.models.XLMRobertaBackbone.from_preset(
        "xlm_roberta_base_multi",
    )
    model(input_data)

    # Randomly initialized XLM-R model with custom config.
    model = keras_hub.models.XLMRobertaBackbone(
        vocabulary_size=250002,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128
    )
    model(input_data)
    ```
    """
