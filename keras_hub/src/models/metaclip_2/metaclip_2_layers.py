"""MetaCLIP 2 layer implementations."""

import math

from keras import layers
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


def quick_gelu(x):
    """QuickGELU activation function.

    This is an approximation of GELU that is used by default in MetaCLIP 2.
    """
    return x * ops.sigmoid(1.702 * x)


class MetaCLIP2VisionEmbedding(layers.Layer):
    """Vision embedding layer for MetaCLIP 2.

    Args:
        hidden_dim: int. The dimensionality of the embedding.
        patch_size: int. The size of each patch.
        image_size: int. The size of the input image (assumed square).
        data_format: str. The data format ("channels_first" or "channels_last").
        dtype: The dtype of the layer weights.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_size,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        data_format = standardize_data_format(data_format)
        self.data_format = data_format
        num_patches = (image_size // patch_size) ** 2
        self.num_positions = num_patches + 1

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            num_patches + 1, hidden_dim, dtype=dtype, name="position_embedding"
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=(self.hidden_dim,),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="class_embedding",
        )
        self.position_ids = self.add_weight(
            shape=(1, self.num_positions),
            initializer="zeros",
            dtype=int,
            trainable=False,
            name="position_ids",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding.build(self.position_ids.shape)

    def call(self, inputs, training=None):
        x = inputs
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.patch_embedding(x, training=training)
        if self.data_format == "channels_last":
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, -1, self.hidden_dim)
            )
        else:
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, self.hidden_dim, -1)
            )
            patch_embeddings = ops.transpose(patch_embeddings, (0, 2, 1))
        class_embeddings = ops.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeddings = ops.tile(class_embeddings, (batch_size, 1, 1))
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = ops.concatenate(
            [class_embeddings, patch_embeddings], axis=1
        )
        return ops.add(embeddings, position_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        return output_shape


class MetaCLIP2EncoderLayer(layers.Layer):
    """Transformer encoder layer for MetaCLIP 2.

    Args:
        hidden_dim: int. The hidden dimension of the layer.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The intermediate dimension for the MLP.
        intermediate_activation: str or callable. The activation function.
            Defaults to "quick_gelu".
        use_causal_mask: bool. Whether to use causal masking.
    """

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
        self.attention = layers.MultiHeadAttention(
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


class MetaCLIP2Head(layers.Layer):
    """Head layer for MetaCLIP 2.

    Takes `vision_embedding` and `text_embedding` as inputs to compute the
    corresponding logits. Both embeddings are L2 normalized and used to compute
    pairwise cosine similarity. The resulting logits are then scaled by a
    learnable `logit_scale` parameter.

    Call arguments:
        vision_embedding: A tensor of shape `(batch_size, hidden_dim)`.
        text_embedding: A tensor of shape `(batch_size, hidden_dim)`.
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=lambda *a, **kw: math.log(1 / 0.07),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )

    def call(self, vision_embedding, text_embedding):
        normalized_vision_embedding = ops.sqrt(
            ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
        )
        normalized_text_embedding = ops.sqrt(
            ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True)
        )
        vision_embedding = vision_embedding / normalized_vision_embedding
        text_embedding = text_embedding / normalized_text_embedding
        logit_scale = ops.exp(self.logit_scale)
        text_logits = (
            ops.matmul(
                text_embedding,
                ops.transpose(vision_embedding),
            )
            * logit_scale
        )
        vision_logits = ops.transpose(text_logits)
        return vision_logits, text_logits

    def compute_output_shape(
        self, vision_embedding_shape, text_embedding_shape
    ):
        vision_logits_shape = (
            vision_embedding_shape[0],
            text_embedding_shape[0],
        )
        text_logits_shape = (
            text_embedding_shape[0],
            vision_embedding_shape[0],
        )
        return vision_logits_shape, text_logits_shape
