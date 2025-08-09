import keras
from keras import ops

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.vit.vit_layers import MLP


class PARSeqDecoderBlock(keras.layers.Layer):
    """A decoder block for the PARSeq model.

    This block consists of self-attention, cross-attention, and a multilayer
    perceptron (MLP). It also includes layer normalization and dropout layers.

    Args:
        hidden_dim: int. The dimension of the hidden layers.
        num_heads: int. The number of attention heads.
        mlp_dim: int. The dimension of the MLP hidden layer.
        dropout_rate: float. The dropout rate used in the feedforward layers.
        attention_dropout: float. The dropout rate for the attention weights.
        layer_norm_epsilon: float. A small float added to the denominator for
            numerical stability in layer normalization.
        **kwargs: Additional keyword arguments passed to the base
            `keras.layers.Layer` constructor.
    """

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
        self.query_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="query_layer_norm",
            dtype=self.dtype_policy,
        )
        self.query_layer_norm.build(input_shape)
        self.content_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="content_layer_norm",
            dtype=self.dtype_policy,
        )
        self.content_layer_norm.build(input_shape)
        self.self_attention = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.attention_dropout,
            name="self_attention",
            dtype=self.dtype_policy,
        )
        self.self_attention.build(input_shape, input_shape)
        self.cross_attention = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.attention_dropout,
            name="cross_attention",
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
        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="decoder_block_dropout",
        )

        self.built = True

    def forward_stream(
        self,
        target,
        target_norm,
        target_kv,
        memory,
        padding_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=0,
        train_attention_mask=None,
    ):
        self_attention_new_cache = None
        if train_attention_mask is None:
            target_attention_mask = self._compute_attention_mask(
                target_norm,
                padding_mask,
                self_attention_cache,
                self_attention_cache_update_index,
            )
        else:
            target_attention_mask = merge_padding_and_attention_mask(
                target_norm, padding_mask, attention_mask=train_attention_mask
            )

        if self_attention_cache is not None:
            target2, self_attention_new_cache = self.self_attention(
                target_norm,
                target_kv,
                target_kv,
                attention_mask=target_attention_mask,
                cache=self_attention_cache,
                cache_update_index=self_attention_cache_update_index,
            )
        else:
            target2 = self.self_attention(
                target_norm,
                target_kv,
                target_kv,
                attention_mask=target_attention_mask,
            )
        target = ops.add(target, self.dropout(target2))
        target2 = self.cross_attention(
            self.layer_norm_1(target),
            memory,
            memory,
        )
        target = ops.add(target, self.dropout(target2))

        target2 = self.mlp(self.layer_norm_2(target))
        target = ops.add(target, target2)

        return target, self_attention_new_cache

    def call(
        self,
        query,
        content,
        memory,
        padding_mask=None,
        update_content=True,
        query_self_attention_cache=None,
        query_self_attention_cache_update_index=0,
        content_self_attention_cache=None,
        content_self_attention_cache_update_index=0,
        query_mask=None,
        content_mask=None,
    ):
        # position + token embeddings
        query_norm = self.query_layer_norm(query)
        # position embeddings
        content_norm = self.content_layer_norm(content)
        (
            query,
            query_self_attention_new_cache,
        ) = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            padding_mask=padding_mask,
            train_attention_mask=query_mask,
            self_attention_cache=query_self_attention_cache,
            self_attention_cache_update_index=query_self_attention_cache_update_index,
        )

        if update_content:
            (
                content,
                content_self_attention_new_cache,
            ) = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,  # image embeddings (encoder embeddings)
                padding_mask=padding_mask,
                train_attention_mask=content_mask,
                self_attention_cache=content_self_attention_cache,
                self_attention_cache_update_index=content_self_attention_cache_update_index,
            )

        return_values = [query, content]

        if query_self_attention_cache is not None:
            return_values.append(query_self_attention_new_cache)
        if update_content and content_self_attention_cache is not None:
            return_values.append(content_self_attention_new_cache)
        elif not update_content and content_self_attention_cache is not None:
            return_values.append(content_self_attention_cache)

        return tuple(return_values)

    def _compute_attention_mask(
        self, x, padding_mask, cache, cache_update_index
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )
        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

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
    """The PARSeq decoder.

    This decoder consists of multiple decoder blocks and a token embedding
    layer. It takes token IDs and memory from the encoder as input and outputs a
    sequence of hidden states.

    Args:
        vocabulary_size: int. The size of the vocabulary.
        max_label_length: int. The maximum length of the label sequence.
        num_layers: int. The number of decoder layers.
        hidden_dim: int. The dimension of the hidden layers.
        mlp_dim: int. The dimension of the MLP hidden layer.
        num_heads: int. The number of attention heads.
        dropout_rate: float. The dropout rate.
        attention_dropout: float. The dropout rate for the attention weights.
        layer_norm_epsilon: float. A small float added to the denominator for
            numerical stability in layer normalization.
        **kwargs: Additional keyword arguments passed to the base
            `keras.layers.Layer` constructor.
    """

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
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_layers = num_layers

    def build(self, input_shape):
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
            dtype=self.dtype,
        )
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="decoder_dropout"
        )
        self.decoder_layers = []
        for i in range(self.num_layers):
            decoder_layer = PARSeqDecoderBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"decoder_layer_{i}",
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
        token_ids,
        memory,
        padding_mask=None,
        query_mask=None,
        content_mask=None,
    ):
        bs, tokens_length = ops.shape(token_ids)
        # <bos> stands for the null context. We only supply position information
        # for characters after <bos>.
        null_context = self.hidden_dim**0.5 * self.token_embedding(
            token_ids[:, :1]
        )
        if tokens_length > 1:
            content = self.pos_query_embeddings[:, : tokens_length - 1, :]
            content = content + self.hidden_dim**0.5 * self.token_embedding(
                token_ids[:, 1:]
            )
            content = ops.concatenate([null_context, content], axis=1)
        else:
            content = null_context

        content = self.dropout(content)

        query = ops.multiply(
            ops.ones((bs, 1, 1), dtype=self.dtype),
            self.pos_query_embeddings[:, :tokens_length, :],
        )
        query = self.dropout(query)

        for i, decoder_layer in enumerate(self.decoder_layers):
            last = i == self.num_layers - 1
            query, content = decoder_layer(
                query=query,
                content=content,
                memory=memory,
                padding_mask=padding_mask,
                update_content=not last,
                query_mask=query_mask,
                content_mask=content_mask,
            )

        query = self.layer_norm(query)

        return query

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
