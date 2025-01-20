from keras import layers

LAYERNORM_EPSILON = 1e-5


class DecoderLayer(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim=2048,
        dropout=0.1,
        activation="gelu",
        name="decoderlayer",
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name=f"{name}_sattn",
        )
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name=f"{name}_xattn",
        )
        self.linear1 = layers.Dense(
            mlp_dim, activation=activation, name=f"{name}_dense1"
        )
        self.dropout = layers.Dropout(dropout, name=f"{name}_dropout")
        self.linear2 = layers.Dense(embed_dim, name=f"{name}_dense2")
        self.norm1 = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm1"
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm2"
        )
        self.norm_q = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_normq"
        )
        self.norm_c = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_normc"
        )
        self.dropout1 = layers.Dropout(dropout, name=f"{name}_dropout1")
        self.dropout2 = layers.Dropout(dropout, name=f"{name}_dropout2")
        self.dropout3 = layers.Dropout(dropout, name=f"{name}_dropout3")

    def forward_stream(
        self,
        tokens,
        token_norm,
        token_kv,
        memory,
        token_mask,
        token_key_padding_mask,
    ):
        if token_key_padding_mask is not None:
            # ignore padded tokens
            token_mask = (token_mask[None, :, :] == 1.0) & (
                token_key_padding_mask[:, None, :] == 0.0
            )

        tokens2, sa_weights = self.self_attn(
            token_norm,
            token_kv,
            token_kv,
            return_attention_scores=True,
            attention_mask=token_mask,
        )
        tokens2 = self.dropout1(tokens2)
        tokens = tokens + tokens2

        tokens1_norm = self.norm1(tokens)
        tokens2, ca_weights = self.cross_attn(
            tokens1_norm, memory, memory, return_attention_scores=True
        )
        tokens2 = self.dropout2(tokens2)
        tokens = tokens + tokens2

        tokens2 = self.linear1(self.norm2(tokens))
        tokens2 = self.dropout(tokens2)
        tokens2 = self.linear2(tokens2)
        tokens2 = self.dropout3(tokens2)
        tokens = tokens + tokens2
        return tokens, sa_weights, ca_weights

    def call(
        self,
        query,
        content,
        memory,
        query_mask=None,
        content_mask=None,
        content_key_padding_mask=None,
        update_content=True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            query_mask,
            content_key_padding_mask,
        )[0]
        if update_content:
            content = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,
                content_mask,
                content_key_padding_mask,
            )[0]
        return query, content


class Decoder(layers.Layer):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        mlp_dim=2048,
        dropout=0.1,
        activation="gelu",
        name="decoder",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)
        self.num_layers = num_layers
        self.layers = [
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                activation=activation,
                name=f"{name}_layer{i}",
            )
            for i in range(num_layers)
        ]
        self.norm = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm"
        )

    def call(
        self,
        query,
        content,
        memory,
        query_mask=None,
        content_mask=None,
        content_key_padding_mask=None,
        training=False,
    ):
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = layer(
                query,
                content,
                memory,
                query_mask=query_mask,
                content_mask=content_mask,
                content_key_padding_mask=content_key_padding_mask,
                update_content=not last,
                training=training,
            )
        query = self.norm(query)
        return query


# class TokenEmbedding(layers.Layer):
#     def __init__(self, charset_size, embed_dim, name="embedding", **kwargs):
#         super().__init__(**kwargs, name=name)
#         self.embed_dim = embed_dim
#         self.embedding = layers.Embedding(
#             input_dim=charset_size, output_dim=embed_dim, name=f"{name}_embed"
#         )

#     def call(self, tokens):
#         # tokens is shape (B, S)
#         out = self.embedding(tokens)
#         out = self.embed_dim**0.5 * out
#         return out
