import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras.saving.register_keras_serializable(package="keras_hub")
class QFormerAttention(keras.layers.Layer):
    """Single attention block (self or cross) with post-LN residual."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        kv_dim,
        layer_norm_epsilon,
        dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.kv_dim = kv_dim if kv_dim is not None else hidden_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, inputs_shape):
        # inputs_shape is either a single shape or [query_shape, kv_shape]
        if isinstance(inputs_shape, (list, tuple)) and isinstance(
            inputs_shape[0], (list, tuple)
        ):
            query_shape, kv_shape = inputs_shape[0], inputs_shape[1]
        else:
            query_shape = inputs_shape
            kv_shape = inputs_shape
        self.attention.build(query_shape, kv_shape)
        self.layer_norm.build(query_shape)
        super().build(inputs_shape)

    def call(self, inputs, attention_mask=None, training=None):
        if isinstance(inputs, (list, tuple)):
            query, key_value = inputs[0], inputs[1]
        else:
            query, key_value = inputs, inputs

        attn = self.attention(
            query,
            key_value,
            key_value,
            attention_mask=attention_mask,
            training=training,
        )
        attn = self.dropout_layer(attn, training=training)
        return self.layer_norm(query + attn)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            return input_shape[0]
        return input_shape

    def compute_output_spec(self, inputs):
        query = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        return keras.KerasTensor(query.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "kv_dim": self.kv_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class BLIP2QFormerTextEmbeddings(keras.layers.Layer):
    """Instruction-text embeddings for the InstructBLIP Q-Former.

    Maps Q-Former instruction token ids to embeddings using a BERT-style
    word-embedding table plus learned absolute position embeddings. The shared
    embeddings LayerNorm and dropout are applied by the parent `BLIP2QFormer`
    over the concatenated `[query_tokens, text_embeddings]` sequence, matching
    HuggingFace's `InstructBlipQFormerEmbeddings`, so this layer only returns
    the summed (pre-LayerNorm) text embeddings.

    Args:
        vocabulary_size: int. Q-Former text vocabulary size (BERT, 30522).
        hidden_dim: int. Embedding dimensionality (Q-Former hidden size).
        max_position_embeddings: int. Maximum instruction sequence length.
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        max_position_embeddings,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings

        self.word_embeddings = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            dtype=self.dtype_policy,
            name="word_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=max_position_embeddings,
            output_dim=hidden_dim,
            dtype=self.dtype_policy,
            name="position_embeddings",
        )

    def build(self, input_shape):
        self.word_embeddings.build(input_shape)
        self.position_embeddings.build((None, None))
        super().build(input_shape)

    def call(self, token_ids):
        word_embeds = self.word_embeddings(token_ids)
        seq_len = ops.shape(token_ids)[-1]
        position_ids = ops.expand_dims(
            ops.arange(seq_len, dtype="int32"), axis=0
        )
        pos_embeds = self.position_embeddings(position_ids)
        return word_embeds + ops.cast(pos_embeds, word_embeds.dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.hidden_dim,)

    def compute_output_spec(self, token_ids):
        return keras.KerasTensor(
            token_ids.shape + (self.hidden_dim,), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class QFormerLayer(keras.layers.Layer):
    """One Q-Former transformer block.

    In the functional graph this layer is called as:
        layer([query_tokens, vision_features])
    """

    def __init__(
        self,
        has_cross_attention,
        num_heads,
        hidden_dim,
        intermediate_dim,
        vision_dim,
        layer_norm_epsilon,
        dropout,
        instruction_aware=False,
        num_query_tokens=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_cross_attention = has_cross_attention
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_dim = vision_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.instruction_aware = instruction_aware
        self.num_query_tokens = num_query_tokens

        self.self_attention = QFormerAttention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            kv_dim=hidden_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        if has_cross_attention:
            self.cross_attention = QFormerAttention(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                kv_dim=vision_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                dtype=self.dtype_policy,
                name="cross_attention",
            )
        self.intermediate_dense = keras.layers.Dense(
            intermediate_dim,
            activation="gelu",
            use_bias=True,
            dtype=self.dtype_policy,
            name="intermediate_dense",
        )
        self.output_dense = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="output_dense",
        )
        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="output_layer_norm",
        )
        self.output_dropout = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy, name="output_dropout"
        )

        # InstructBLIP keeps a *second*, independent feed-forward network for
        # the instruction-text tokens (HF `intermediate` / `output`), distinct
        # from the query feed-forward above (HF `intermediate_query` /
        # `output_query`). It is only built when the layer is instruction-aware.
        if self.instruction_aware:
            self.text_intermediate_dense = keras.layers.Dense(
                intermediate_dim,
                activation="gelu",
                use_bias=True,
                dtype=self.dtype_policy,
                name="text_intermediate_dense",
            )
            self.text_output_dense = keras.layers.Dense(
                hidden_dim,
                use_bias=True,
                dtype=self.dtype_policy,
                name="text_output_dense",
            )
            self.text_output_layer_norm = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="text_output_layer_norm",
            )
            self.text_output_dropout = keras.layers.Dropout(
                dropout, dtype=self.dtype_policy, name="text_output_dropout"
            )

    def build(self, inputs_shape):
        # When called from the functional graph with
        # [query_tokens, vision_input], Keras passes inputs_shape as a list of
        # two shapes.
        if isinstance(inputs_shape, (list, tuple)) and isinstance(
            inputs_shape[0], (list, tuple)
        ):
            query_shape = inputs_shape[0]
            vision_shape = inputs_shape[1] if len(inputs_shape) > 1 else None
        else:
            query_shape = inputs_shape
            vision_shape = None

        self.self_attention.build([query_shape, query_shape])
        if self.has_cross_attention and vision_shape is not None:
            self.cross_attention.build([query_shape, vision_shape])
        self.intermediate_dense.build(query_shape)

        ffn_out_shape = query_shape[:-1] + (self.intermediate_dim,)
        self.output_dense.build(ffn_out_shape)
        self.output_layer_norm.build(query_shape)

        if self.instruction_aware:
            self.text_intermediate_dense.build(query_shape)
            self.text_output_dense.build(ffn_out_shape)
            self.text_output_layer_norm.build(query_shape)
        super().build(inputs_shape)

    def call(self, inputs, training=None):
        # Vision-only graph passes [hidden_tokens, vision_features].
        # Instruction-aware graph passes
        # [hidden_tokens, vision_features, attention_mask], where
        # `hidden_tokens` is the concatenated [query; instruction] sequence.
        attention_mask = None
        if isinstance(inputs, (list, tuple)):
            hidden_tokens, vision_features = inputs[0], inputs[1]
            if len(inputs) > 2:
                attention_mask = inputs[2]
        else:
            hidden_tokens, vision_features = inputs, None

        x = self.self_attention(
            [hidden_tokens, hidden_tokens],
            attention_mask=attention_mask,
            training=training,
        )

        if not self.instruction_aware:
            if self.has_cross_attention:
                x = self.cross_attention(
                    [x, vision_features], training=training
                )
            h = self.intermediate_dense(x)
            h = self.output_dense(h)
            h = self.output_dropout(h, training=training)
            return self.output_layer_norm(x + h)

        # Instruction-aware: split into query / instruction-text parts. Only the
        # query tokens cross-attend to the image and run the query FFN; the
        # instruction tokens run their own FFN. Both halves are recombined so
        # later self-attention layers still see the full sequence.
        query_part = x[:, : self.num_query_tokens, :]
        text_part = x[:, self.num_query_tokens :, :]

        if self.has_cross_attention:
            query_part = self.cross_attention(
                [query_part, vision_features], training=training
            )
        hq = self.intermediate_dense(query_part)
        hq = self.output_dense(hq)
        hq = self.output_dropout(hq, training=training)
        query_out = self.output_layer_norm(query_part + hq)

        ht = self.text_intermediate_dense(text_part)
        ht = self.text_output_dense(ht)
        ht = self.text_output_dropout(ht, training=training)
        text_out = self.text_output_layer_norm(text_part + ht)

        return ops.concatenate([query_out, text_out], axis=1)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            return input_shape[0]
        return input_shape

    def compute_output_spec(self, inputs):
        query = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        return keras.KerasTensor(query.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "has_cross_attention": self.has_cross_attention,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "vision_dim": self.vision_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "instruction_aware": self.instruction_aware,
                "num_query_tokens": self.num_query_tokens,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class BLIP2QueryTokens(keras.layers.Layer):
    """Holds the learned query-token weight and broadcasts over the batch.

    This thin wrapper exists so that the query_tokens parameter lives inside
    a proper Keras layer and participates cleanly in the functional graph.
    """

    def __init__(self, num_query_tokens, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.query_tokens = self.add_weight(
            shape=(1, self.num_query_tokens, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="query_tokens",
        )
        super().build(input_shape)

    def call(self, vision_features):
        batch_size = ops.shape(vision_features)[0]
        return ops.broadcast_to(
            self.query_tokens,
            (batch_size, self.num_query_tokens, self.hidden_dim),
        )

    def compute_output_shape(self, input_shape):
        # input_shape = (batch, num_patches, vision_dim)
        # output      = (batch, num_query_tokens, hidden_dim)
        return (input_shape[0], self.num_query_tokens, self.hidden_dim)

    def compute_output_spec(self, vision_features):
        output_shape = (
            vision_features.shape[0],
            self.num_query_tokens,
            self.hidden_dim,
        )
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_tokens": self.num_query_tokens,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras_hub_export("keras_hub.models.BLIP2QFormer")
class BLIP2QFormer(keras.Model):
    """Querying Transformer (Q-Former) for BLIP-2.

    Functional Keras model — auto-builds on instantiation, so `.summary()`
    works without a manual `.build()` call.

    Args:
        num_query_tokens: int. Number of learnable query tokens.
        num_layers: int. Number of transformer layers.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Transformer hidden size.
        intermediate_dim: int. FFN inner dimension.
        vision_dim: int. Dimension of input visual features.
        cross_attention_frequency: int. Insert cross-attention every N layers.
        dropout: float. Dropout probability.
        layer_norm_epsilon: float. LayerNorm epsilon.
        instruction_aware: bool. When `True` (InstructBLIP), the Q-Former also
            consumes an instruction (`qformer_token_ids`/`qformer_padding_mask`)
            that is embedded and concatenated with the query tokens so the
            queries extract instruction-conditioned visual features. Defaults to
            `False` (BLIP-2 behavior: vision features only).
        qformer_vocabulary_size: int or None. Vocabulary size of the Q-Former
            instruction tokenizer (BERT, 30522). Required when
            `instruction_aware` is `True`.
        max_position_embeddings: int. Maximum instruction length for the
            Q-Former text position embeddings. Defaults to `512`.
    """

    def __init__(
        self,
        num_query_tokens,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        vision_dim,
        cross_attention_frequency,
        dropout,
        layer_norm_epsilon,
        instruction_aware=False,
        qformer_vocabulary_size=None,
        max_position_embeddings=512,
        name=None,
        **kwargs,
    ):
        if instruction_aware and qformer_vocabulary_size is None:
            raise ValueError(
                "`qformer_vocabulary_size` must be set when "
                "`instruction_aware=True`."
            )
        text_embeddings = None
        if instruction_aware:
            text_embeddings = BLIP2QFormerTextEmbeddings(
                vocabulary_size=qformer_vocabulary_size,
                hidden_dim=hidden_dim,
                max_position_embeddings=max_position_embeddings,
                dtype=kwargs.get("dtype", None),
                name="text_embeddings",
            )

        query_tokens_layer = BLIP2QueryTokens(
            num_query_tokens=num_query_tokens,
            hidden_dim=hidden_dim,
            dtype=kwargs.get("dtype", None),
            name="query_tokens_layer",
        )
        layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=kwargs.get("dtype", None),
            name="layer_norm",
        )
        dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=kwargs.get("dtype", None),
            name="dropout",
        )
        transformer_layers = [
            QFormerLayer(
                has_cross_attention=(i % cross_attention_frequency == 0),
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                vision_dim=vision_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                instruction_aware=instruction_aware,
                num_query_tokens=num_query_tokens,
                dtype=kwargs.get("dtype", None),
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

        vision_input = keras.Input(
            shape=(None, vision_dim), name="vision_features"
        )

        query_tokens = query_tokens_layer(vision_input)

        if instruction_aware:
            qformer_token_ids = keras.Input(
                shape=(None,), dtype="int32", name="qformer_token_ids"
            )
            qformer_padding_mask = keras.Input(
                shape=(None,), dtype="int32", name="qformer_padding_mask"
            )
            text_embeds = text_embeddings(qformer_token_ids)
            hidden = ops.concatenate([query_tokens, text_embeds], axis=1)
            hidden = layer_norm(hidden)
            hidden = dropout_layer(hidden)

            # Self-attention key padding mask over [query; instruction]: query
            # tokens are always attended to, instruction tokens follow their
            # padding mask. Shape (B, 1, seq) broadcasts over heads/queries.
            query_mask = ops.ones_like(query_tokens[..., 0])
            full_mask = ops.concatenate(
                [query_mask, ops.cast(qformer_padding_mask, query_mask.dtype)],
                axis=1,
            )
            attention_mask = ops.cast(full_mask[:, None, :], "bool")

            for t_layer in transformer_layers:
                hidden = t_layer([hidden, vision_input, attention_mask])

            # Only the query-token outputs feed the language model.
            outputs = hidden[:, :num_query_tokens, :]
            inputs = {
                "vision_features": vision_input,
                "qformer_token_ids": qformer_token_ids,
                "qformer_padding_mask": qformer_padding_mask,
            }
        else:
            query_tokens = layer_norm(query_tokens)
            query_tokens = dropout_layer(query_tokens)
            for t_layer in transformer_layers:
                query_tokens = t_layer([query_tokens, vision_input])
            outputs = query_tokens
            inputs = vision_input

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name,
            **kwargs,
        )

        self.text_embeddings = text_embeddings
        self.instruction_aware = instruction_aware
        self.qformer_vocabulary_size = qformer_vocabulary_size
        self.max_position_embeddings = max_position_embeddings
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_dim = vision_dim
        self.cross_attention_frequency = cross_attention_frequency
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.query_tokens_layer = query_tokens_layer
        self.layer_norm = layer_norm
        self.transformer_layers = transformer_layers

    @property
    def query_tokens(self):
        return self.query_tokens_layer.query_tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_tokens": self.num_query_tokens,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "vision_dim": self.vision_dim,
                "cross_attention_frequency": self.cross_attention_frequency,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "instruction_aware": self.instruction_aware,
                "qformer_vocabulary_size": self.qformer_vocabulary_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config
