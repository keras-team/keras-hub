import keras
from keras import ops

from keras_hub.src.models.sam.sam_layers import (
    MultiHeadAttentionWithDownsampling,
)
from keras_hub.src.models.sam.sam_layers import TwoWayMultiHeadAttention


class TwoWayTransformer(keras.layers.Layer):
    """A two-way cross-attention transformer decoder.

    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.
    The transformer decoder design is shown in
    [1](https://arxiv.org/abs/2304.02643).
    Each decoder layer performs 4 steps:
    (1) self-attention on the tokens,
    (2) cross-attention from tokens (as queries) to the image embedding,
    (3) a point-wise MLPupdates each token, and
    (4) cross-attention from the image embedding (as
    queries) to tokens. This last step updates the image embedding with prompt
    information. Each self/cross-attention and MLP has a residual connection
    and layer normalization.
    To ensure the decoder has access to critical geometric information the
    positional encodings are added to the image embedding whenever they
    participate in an attention layer. Additionally, the entire original
    prompt tokens (including their positional encodings) are re-added to the
    updated tokens whenever they participate in an attention layer. This
    allows for a strong dependence on both the prompt token's geometric
    location and type.

    Args:
        num_layers: int, optional. The num_layers of the attention blocks
            (the number of attention blocks to use). Defaults to `2`.
        hidden_size: int, optional. The number of features of the input image
            and point embeddings. Defaults to `256`.
        num_heads: int, optional. Number of heads to use in the attention
            layers. Defaults to `8`.
        intermediate_dim: int, optional. The number of units in the hidden
            layer of the MLP block used in the attention layers.
            Defaults to `2048`.
        activation: str, optional. The activation of the MLP block's output
            layer used in the attention layers. Defaults to `"relu"`.
        attention_downsample_rate: int, optional. The downsample rate of the
            attention layers. Defaults to `2`.
    """

    def __init__(
        self,
        *,
        num_layers=2,
        hidden_size=256,
        num_heads=8,
        intermediate_dim=2048,
        activation="relu",
        attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.attention_downsample_rate = attention_downsample_rate
        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                TwoWayMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=hidden_size // num_heads,
                    intermediate_dim=intermediate_dim,
                    skip_first_layer_pos_embedding=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                    dtype=self.dtype_policy,
                )
            )
        self.final_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=hidden_size // num_heads,
                downsample_rate=attention_downsample_rate,
                dtype=self.dtype_policy,
            )
        )
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        for layer in self.layers:
            layer.build()
        self.final_attention_token_to_image.build()
        self.final_layer_norm.build([None, None, self.hidden_size])
        self.built = True

    def call(
        self, image_embedding, image_positional_embeddings, point_embedding
    ):
        shape = ops.shape(image_embedding)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        image_embedding = ops.reshape(image_embedding, (B, H * W, C))

        shape = ops.shape(image_positional_embeddings)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        image_positional_embeddings = ops.reshape(
            image_positional_embeddings, (B, H * W, C)
        )
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pos_embedding=point_embedding,
                key_pos_embedding=image_positional_embeddings,
            )

        queries_with_pos_embedding = queries + point_embedding
        keys_with_pos_embedding = keys + image_positional_embeddings
        attention_map = self.final_attention_token_to_image(
            query=queries_with_pos_embedding,
            key=keys_with_pos_embedding,
            value=keys,
        )
        queries = queries + attention_map
        queries = self.final_layer_norm(queries)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
                "attention_downsample_rate": self.attention_downsample_rate,
            }
        )
        return config
