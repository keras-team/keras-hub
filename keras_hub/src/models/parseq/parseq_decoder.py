import keras

from keras_hub.src.models.vit.vit_layers import MLP


class PARSeqDecoderBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        key_dim = hidden_dim // num_heads

        # === Config ===
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        self.layer_norm_q = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="layer_norm_q",
            dtype=self.dtype_policy,
        )
        self.layer_norm_q.build(input_shape)
        self.layer_norm_kv = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="layer_norm_kv",
            dtype=self.dtype_policy,
        )
        self.layer_norm_kv.build(input_shape)
        self.self_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.attention_dropout,
            name="self_attention",
            dtype=self.dtype_policy,
        )
        self.self_attention.build(input_shape, input_shape)
        self.cross_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.attention_dropout,
            name="corss_attention",
            dtype=self.dtype_policy,
        )
        self.cross_attention.build(input_shape, input_shape)

        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_1",
            dtype=self.dtype_policy,
        )
        self.layer_norm_1.build((None, None, self.hidden_dim))
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_2",
            dtype=self.dtype_policy,
        )
        self.layer_norm_2.build((None, None, self.hidden_dim))
        self.mlp = MLP(
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self.mlp.build((None, None, self.hidden_dim))
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        self.built = True

    def forward_stream(
        self,
        target,
        target_norm,
        target_kv,
        memory,
        target_mask=None,
        target_key_padding_mask=None,
    ):
        target_mask = None
        if target_mask is not None:
            # ignore padded tokens
            target_mask = (target_mask[None, :, :] == 1.0) & (
                target_key_padding_mask[:, None, :] == 0.0
            )
        target2, sa_weights = self.self_attention(
            target_norm,
            target_kv,
            target_kv,
            attention_mask=target_mask,
            return_attention_scores=True,
        )
        target = target + self.dropout(target2)

        target2, ca_weights = self.cross_attention(
            self.layer_norm_1(target),
            memory,
            memory,
            return_attention_scores=True,
        )
        target = target + self.dropout(target2)

        target2 = self.mlp(self.layer_norm_2(target))
        target = target + target2
        return target, sa_weights, ca_weights

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
        query_norm = self.layer_norm_q(query)
        content_norm = self.layer_norm_kv(content)
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "key_dim": self.key_dim,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class PARSeqDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        mlp_dim,
        num_heads,
        dropout_rate=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_layers = num_layers

    def build(self, input_shape):
        self.decoder_layers = []
        for _ in range(self.num_layers):
            decoder_layer = PARSeqDecoderBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
            )
            decoder_layer.build((None, None, self.hidden_dim))
            self.decoder_layers.append(decoder_layer)

        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.layer_norm.build((None, None, self.hidden_dim))
        self.built = True

    def call(
        self,
        query,
        content,
        memory,
        query_mask=None,
        content_mask=None,
        content_key_padding_mask=None,
    ):
        for i, decoder_layer in enumerate(self.decoder_layers):
            last = i == self.num_layers - 1
            query, content = decoder_layer(
                query=query,
                content=content,
                memory=memory,
                query_mask=query_mask,
                content_mask=content_mask,
                content_key_padding_mask=content_key_padding_mask,
                update_content=not last,
            )

        query = self.layer_norm(query)

        return query

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class PARSeqDecode(keras.layers.Layer):
    def __init__(
        self,
        vocabulary_size,
        max_label_length,
        num_layers,
        hidden_dim,
        mlp_dim,
        num_heads,
        dropout_rate=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.max_label_length = max_label_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        self.decoder = PARSeqDecoder(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        self.decoder.build(input_shape)

        self.token_embedding = keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="token_embedding",
        )
        self.token_embedding.build((1, self.vocabulary_size))
        self.pos_query_embeddings = self.add_weight(
            shape=(1, self.max_label_length + 1, self.hidden_dim),
            name="pos_query_embeddings",
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        target,
        memory,
        target_mask=None,
        target_padding_mask=None,
        target_query=None,
        target_query_mask=None,
    ):
        N, L = keras.ops.shape(target)
        # <bos> stands for the null context. We only supply position information
        # for characters after <bos>.
        null_ctx = self.hidden_dim**0.5 * self.token_embedding(target[:, :1])
        target_emb = self.pos_query_embeddings[
            :, : L - 1
        ] + self.hidden_dim**0.5 * self.token_embedding(target[:, 1:])
        target_emb = self.dropout(
            keras.ops.concatenate([null_ctx, target_emb], axis=1)
        )
        if target_query is None:
            target_query = (
                keras.ops.ones((N, 1, 1)) * self.pos_query_embeddings[:, :L]
            )

        target_query = self.dropout(target_query)
        return self.decoder(
            query=target_query,
            content=target_emb,
            memory=memory,
            query_mask=target_query_mask,
            content_mask=target_mask,
            content_key_padding_mask=target_padding_mask,
        )

    def compute_output_shape(self, input_shape):
        return (None, None, self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "max_label_length": self.max_label_length,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
