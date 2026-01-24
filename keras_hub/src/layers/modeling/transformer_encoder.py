import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer

from keras_hub.src.layers.modeling.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)


@keras_hub_export("keras_hub.layers.TransformerEncoder")
class TransformerEncoder(keras.layers.Layer):
    """Transformer encoder.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    This layer will compute an attention mask, prioritizing explicitly provided
    masks (a `padding_mask` or a custom `attention_mask`) over an implicit Keras
    padding mask (for example, by passing `mask_zero=True` to a
    `keras.layers.Embedding` layer). If both a `padding_mask` and a
    `attention_mask` are provided, they will be combined to determine the final
    mask. See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `keras.layers.MultiHeadAttention` layer.
        dropout: float. the dropout value, shared by
            `keras.layers.MultiHeadAttention` and feedforward network.
            Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer and the intermediate dense layer  are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_hub.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=(10, 64))
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(2, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True

    def build(self, inputs_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=inputs_shape,
                value=inputs_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=inputs_shape,
                value_shape=inputs_shape,
            )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(inputs_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(
        self,
        inputs,
        padding_mask=None,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.
            return_attention_scores: a boolean indicating whether the output
                should be `(attention_output, attention_scores)` if `True` or
                `attention_output` if `False`. Defaults to `False`.

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)

        if return_attention_scores:
            x, attention_scores = self._self_attention_layer(
                query=x,
                value=x,
                attention_mask=self_attention_mask,
                return_attention_scores=return_attention_scores,
                training=training,
            )
        else:
            x = self._self_attention_layer(
                query=x,
                value=x,
                attention_mask=self_attention_mask,
                training=training,
            )

        x = self._self_attention_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        if return_attention_scores:
            return x, attention_scores

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape
