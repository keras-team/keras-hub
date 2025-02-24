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
    Moonshine encoder block.

    Inherits from `keras.layers.Layer` and overrides the `build` method to
    initialize and construct the self-attention and feedforward sublayers
    using `MoonshineMultiHeadAttention` and either `MoonshineSwiGLU` or
    `MoonshineLinearGeLU` depending on the activation function choice.

    Args:
        hidden_dim (int): The dimensionality of the model.
        inner_dim (int): The inner dimensionality for feedforward layers.
        num_heads (int): The number of attention heads.
        ff_mult (int): Multiplicative factor for the feedforward dimension.
        ff_swiglu (bool): Whether to use SwiGLU in the feedforward branch.
        **kwargs: Additional keyword arguments passed to the base layer.

    Example:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_encoder import (
        MoonshineEncoderBlock
    )

    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    inner_dim = 512
    num_heads = 8

    dummy_input = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, hidden_dim).astype("float32")
    )
    dummy_rot_pos_emb = keras.ops.convert_to_tensor(
        np.random.randn(seq_len, hidden_dim // num_heads).astype("float32")
    )

    encoder_block = MoonshineEncoderBlock(
        hidden_dim=hidden_dim,
        inner_dim=inner_dim,
        num_heads=num_heads,
        ff_mult=4,
        ff_swiglu=False
    )
    output = encoder_block(dummy_input, rot_pos_emb=dummy_rot_pos_emb)
    print(output)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        inner_dim,
        num_heads,
        ff_mult=4,
        ff_swiglu=False,
        pad_head_dim_to_multiple_of=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

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
        if ff_swiglu:
            self.feedforward = MoonshineSwiGLU(
                hidden_dim, ff_mult, name="feedforward"
            )
        else:
            self.feedforward = MoonshineLinearGeLU(
                hidden_dim, ff_mult, name="feedforward"
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

    def call(self, inputs, rot_pos_emb, training=None, **kwargs):
        x = inputs

        # Self-attention block with residual connection.
        attention_residual = x
        x = self.self_attention_layer_norm(x)
        x = self.self_attention_layer(
            query=x,
            value=x,
            key=x,
            rot_pos_emb=rot_pos_emb,
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
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineEncoder(keras.Model):
    """
    Moonshine encoder that processes sequences through multiple encoder blocks.

    Inherits from `keras.layers.Layer` and implements full encoder architecture
    consisting of multiple MoonshineEncoderBlock instances. Includes rotary
    positional embeddings and a final layer normalization.

    Args:
        num_layers (int): Number of encoder blocks in the stack.
        hidden_dim (int): The dimensionality of the model.
        inner_dim (int): The inner dimensionality for feedforward layers.
        num_heads (int): The number of attention heads.
        ff_mult (int): Multiplicative factor for the feedforward dimension.
        ff_swiglu (bool): Whether to use SwiGLU in the feedforward branch.
        **kwargs: Additional keyword arguments passed to the base layer.

    Example:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_encoder import (
        MoonshineEncoder
    )

    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    inner_dim = 512
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
        inner_dim=inner_dim,
        num_heads=num_heads,
        ff_mult=4,
        ff_swiglu=False
    )
    output = encoder([dummy_sequence, dummy_seq_length])
    print(output)
    ```
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        inner_dim,
        num_heads,
        ff_mult=4,
        ff_swiglu=False,
        max_position_embeddings=2048,
        pad_head_dim_to_multiple_of=None,
        partial_rotary_factor=0.62,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

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
                inner_dim=inner_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                ff_swiglu=ff_swiglu,
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
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config
