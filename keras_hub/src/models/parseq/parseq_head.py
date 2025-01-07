from keras import layers

LAYERNORM_EPSILON = 1e-5


class DecoderLayer(layers.Layer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model, dropout=dropout
        )
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model, dropout=dropout
        )
        self.linear1 = layers.Dense(dim_feedforward, activation=activation)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        self.norm2 = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        self.norm_q = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        self.norm_c = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

    def forward_stream(
        self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask
    ):
        tgt_norm = self.norm_q(tgt_norm)
        tgt_kv = self.norm_c(tgt_kv)
        memory = self.norm_c(memory)

        if tgt_key_padding_mask is not None:
            tgt_mask = (tgt_mask[None, :, :] == 1.0) & (
                tgt_key_padding_mask[:, None, :] == 0.0
            )

        tgt2, sa_weights = self.self_attn(
            tgt_norm,
            tgt_kv,
            tgt_kv,
            return_attention_scores=True,
            attention_mask=tgt_mask,
        )
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + tgt2

        tgt1_norm = self.norm1(tgt)
        tgt2, ca_weights = self.cross_attn(
            tgt1_norm, memory, memory, return_attention_scores=True
        )
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + tgt2

        tgt2 = self.linear1(self.norm2(tgt))
        tgt2 = self.dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt2 = self.dropout3(tgt2)
        tgt = tgt + tgt2
        return tgt, sa_weights, ca_weights

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
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layers = [
            DecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ]
        self.norm = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)

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


class TokenEmbedding(layers.Layer):
    def __init__(self, charset_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = layers.Embedding(
            input_dim=charset_size, output_dim=embed_dim
        )

    def call(self, tokens):
        # tokens is shape (B, S)
        out = self.embedding(tokens)
        out = self.embed_dim**0.5 * out
        return out
