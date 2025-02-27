import keras

from keras_hub.src.models.moonshine.moonshine_layers import MoonshineArange
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineSwiGLU
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineEncoderBlock(keras.layers.Layer):
    """
    Moonshine encoder block for transformer-based sequence processing.

    Implements a standard transformer encoder block with self-attention and
    feedforward sublayers, including residual connections and layer
    normalization. The implementation uses Moonshine-specific attention and
    feedforward mechanisms.

    Args:
        hidden_dim: int, Dimension of the model's hidden representations
        throughout the block.
        intermediate_dim: int, Dimension used in projections before applying
        non-linearities.
        num_heads: int, Number of attention heads for multi-head attention
        computation.
        feedforward_expansion_factor: int, Multiplier for expanding the
        dimension in the feedforward network. Default is 4.
        use_swiglu_activation: bool, Whether to use SwiGLU activation (True)
        or LinearGeLU (False) in the feedforward sublayer. Default is False.
        pad_head_dim_to_multiple_of: int, Optional value to pad the head
        dimension to a multiple of this value for hardware optimization.
        Default is None.
        **kwargs: Additional keyword arguments passed to the base layer.

    Examples:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_encoder import (
        MoonshineEncoderBlock
    )

    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    intermediate_dim = 512
    num_heads = 8

    dummy_input = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, hidden_dim).astype("float32")
    )
    dummy_rotary_embedding = keras.ops.convert_to_tensor(
        np.random.randn(seq_len, hidden_dim // num_heads).astype("float32")
    )

    encoder_block = MoonshineEncoderBlock(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False
    )
    output = encoder_block(dummy_input, rotary_embedding=dummy_rotary_embedding)
    print(output)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
        pad_head_dim_to_multiple_of=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation

        # Self-attention sublayers.
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.self_attention_layer = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            name="self_attention_layer",
        )
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="self_attention_layer_norm",
        )

        # Feedforward sublayers.
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="feedforward_layer_norm",
        )
        if use_swiglu_activation:
            self.feedforward = MoonshineSwiGLU(
                hidden_dim, feedforward_expansion_factor, name="feedforward"
            )
        else:
            self.feedforward = MoonshineLinearGeLU(
                hidden_dim, feedforward_expansion_factor, name="feedforward"
            )

    def build(self, input_shape):
        super().build(input_shape)
        # Build self-attention branch.
        self.self_attention_layer_norm.build(input_shape)
        self.self_attention_layer.build(input_shape, input_shape)
        # Build feedforward branch.
        self.feedforward_layer_norm.build(input_shape)
        # The feedforward layer expects the last dimension to be 'hidden_dim'.
        feed_forward_input_shape = list(input_shape)
        feed_forward_input_shape[-1] = self.hidden_dim
        self.feedforward.build(tuple(feed_forward_input_shape))

    def call(self, inputs, rotary_embedding, training=None, **kwargs):
        x = inputs

        # Self-attention block with residual connection.
        attention_residual = x
        x = self.self_attention_layer_norm(x)
        x = self.self_attention_layer(
            query=x,
            value=x,
            key=x,
            rotary_embedding=rotary_embedding,
            training=training,
            **kwargs,
        )
        x = x + attention_residual

        # Feedforward block with residual connection.
        ff_residual = x
        x = self.feedforward_layer_norm(x)
        x = self.feedforward(x)
        x = x + ff_residual

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # ==== Config ====
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineEncoder(keras.Model):
    """
    Full Moonshine encoder stack for sequence modeling tasks.

    Combines multiple MoonshineEncoderBlock instances with rotary positional
    embeddings to process input sequences. This encoder architecture forms
    the core of transformer-based Moonshine models.

    Args:
        num_layers: int, Number of encoder blocks stacked sequentially.
        hidden_dim: int, Dimension of hidden representations throughout the
        model.
        intermediate_dim: int, Dimension used in intermediate projections before
        non-linearities are applied.
        num_heads: int, Number of attention heads in each multi-head attention
        layer.
        feedforward_expansion_factor: int, Multiplier that determines the
        expanded dimension in the feedforward networks. Default is 4.
        use_swiglu_activation: bool, Whether to use SwiGLU activation (True) or
        LinearGeLU (False) in the feedforward sublayers. Default is False.
        max_position_embeddings: int, Maximum sequence length supported by the
        positional embeddings. Default is 2048.
        pad_head_dim_to_multiple_of: int, Optional value to pad the head
        dimension to a multiple of this value for hardware optimization.
        Default is None.
        partial_rotary_factor: float, Factor controlling what portion of the
        embedding dimension receives rotary position embeddings. Default is
        0.62.
        **kwargs: Additional keyword arguments passed to the parent Model.

    Examples:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_encoder import (
        MoonshineEncoder
    )

    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    intermediate_dim = 512
    num_heads = 8
    num_layers = 3

    dummy_sequence = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, hidden_dim).astype("float32")
    )
    dummy_seq_length = keras.ops.convert_to_tensor(
        np.array([seq_len, seq_len]).astype("int32")
    )

    encoder = MoonshineEncoder(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False
    )
    output = encoder([dummy_sequence, dummy_seq_length])
    print(output)
    ```
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
        max_position_embeddings=2048,
        pad_head_dim_to_multiple_of=None,
        partial_rotary_factor=0.62,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation

        self.max_position_embeddings = max_position_embeddings
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.partial_rotary_factor = partial_rotary_factor

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.arange = MoonshineArange(name="arange")
        self.rotary_embedding = MoonshineRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            name="rotary_embedding",
        )

        self.encoder_layers = []
        for i in range(num_layers):
            block = MoonshineEncoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                feedforward_expansion_factor=feedforward_expansion_factor,
                use_swiglu_activation=use_swiglu_activation,
                pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
                name=f"moonshine_encoder_block_{i}",
            )
            self.encoder_layers.append(block)

        self.final_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="final_layer_norm",
        )

    def build(self, input_shape):
        super().build(input_shape)
        sequence_shape, _ = input_shape
        self.arange.build(input_shape=(None,))
        self.rotary_embedding.build(input_shape=(None,))
        self.final_layer_norm.build(sequence_shape)
        for layer in self.encoder_layers:
            layer.build(sequence_shape)

    def call(self, inputs, training=None):
        # ==== Functional Model ====
        encoder_sequence, sequence_length = inputs
        pos_indices = self.arange(sequence_length[0])
        pos_emb = self.rotary_embedding(pos_indices)

        x = encoder_sequence
        for block in self.encoder_layers:
            x = block(x, pos_emb, training=training)
        return self.final_layer_norm(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        # ==== Config ====
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
            }
        )
        return config
