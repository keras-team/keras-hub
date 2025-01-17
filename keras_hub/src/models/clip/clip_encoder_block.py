from keras import dtype_policies
from keras import layers
from keras import ops


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


# TODO: Deprecate this in favor of `keras.layers.MultiHeadAttention` once the
# dtype compatibility issue is resolved.
class CLIPMultiHeadAttention(layers.MultiHeadAttention):
    def _masked_softmax(self, attention_scores, attention_mask=None):
        attention_scores = super()._masked_softmax(
            attention_scores, attention_mask
        )
        return ops.cast(attention_scores, self._value_dense.compute_dtype)


class CLIPEncoderBlock(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        use_causal_mask=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`. "
                f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.use_causal_mask = use_causal_mask

        if intermediate_activation == "quick_gelu":
            intermediate_activation = quick_gelu

        self.layer_norm_1 = layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="layer_norm_1"
        )
        self.attention = CLIPMultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm_2 = layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="layer_norm_2"
        )
        self.dense_1 = layers.Dense(
            self.intermediate_dim, dtype=self.dtype_policy, name="dense_1"
        )
        self.activation = layers.Activation(
            intermediate_activation, dtype=self.dtype_policy, name="activation"
        )
        self.dense_2 = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="dense_2"
        )

    def build(self, input_shape):
        self.layer_norm_1.build(input_shape)
        self.attention.build(input_shape, input_shape, input_shape)
        # Before Keras 3.2, there was no setter for `dtype_policy`. Directly
        # assign a `DTypePolicy` instead.
        self.attention._softmax.dtype_policy = dtype_policies.DTypePolicy(
            "float32"
        )
        self.layer_norm_2.build(input_shape)
        self.dense_1.build(input_shape)
        input_shape = self.dense_1.compute_output_shape(input_shape)
        self.dense_2.build(input_shape)

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape

    def call(self, x, training=None):
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(
            x, x, x, training=training, use_causal_mask=self.use_causal_mask
        )
        x = ops.add(residual, x)

        residual = x
        x = self.dense_1(self.layer_norm_2(residual))
        x = self.activation(x)
        x = self.dense_2(x)
        x = ops.add(residual, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "use_causal_mask": self.use_causal_mask,
            }
        )
        return config
