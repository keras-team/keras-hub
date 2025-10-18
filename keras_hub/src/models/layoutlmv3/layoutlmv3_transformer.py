import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import (
    TransformerEncoder,
)


@keras_hub_export("keras_hub.models.LayoutLMv3TransformerLayer")
class LayoutLMv3TransformerLayer(TransformerEncoder):
    """LayoutLMv3 transformer encoder layer.
    
    This layer implements a transformer encoder block for LayoutLMv3, which
    includes multi-head self-attention and a feed-forward network.
    
    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The output dimension of the first Dense layer
            in the feedforward network.
        dropout: float. Dropout probability.
        activation: string or callable. The activation function to use.
        layer_norm_epsilon: float. The epsilon value in layer normalization
            components.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded attention
            layers.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded attention
            layers.
        **kwargs: additional keyword arguments to pass to TransformerEncoder.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        dropout=0.1,
        activation="gelu",
        layer_norm_epsilon=1e-12,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )
        self.hidden_dim = hidden_dim

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config 