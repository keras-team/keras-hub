import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.parseq.parseq_head import Decoder
from keras_hub.src.models.parseq.parseq_head import TokenEmbedding
from keras_hub.src.models.parseq.parseq_vit import VisionTransformer


@keras_hub_export("keras_hub.models.ParseQBackbone")
class ParseQBackbone(Backbone):
    def __init__(
        self,
        out_channels,
        img_size=(32, 128),
        patch_size=(4, 8),
        in_channels=3,
        embed_dim=384,
        enc_depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        max_text_length=25,
        dec_num_heads=12,
        dec_mlp_ratio=4.0,
        dec_depth=1,
        decode_ar=True,
        refine_iters=1,
        dropout=0.1,
        **kwargs,
    ):
        encoder = VisionTransformer(
            img_size,
            patch_size,
            in_channels,
            None,
            embed_dim,
            enc_depth,
            num_heads,
            mlp_ratio,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
        )

        # Configure the decoder layer parameters
        d_model = embed_dim
        dim_feedforward = int(embed_dim * dec_mlp_ratio)

        # Create the Decoder
        decoder = Decoder(
            num_layers=dec_depth,
            d_model=d_model,
            nhead=dec_num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )

        # Embedding layer for input tokens
        text_embed = TokenEmbedding(out_channels, embed_dim)

        dropout_layer = layers.Dropout(dropout)
        # Output head to project decoder outputs to token probabilities
        dense_head = layers.Dense(out_channels - 2, name="head")

        super().__init__(**kwargs)

        # Positional queries, initialized here and set in build() or after init
        pos_queries = self.add_weight(
            name="pos_queries",
            shape=(1, max_text_length + 1, embed_dim),
            initializer="zeros",
            trainable=True,
        )

        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.bos_id = out_channels - 2
        self.eos_id = 0
        self.pad_id = out_channels - 1
        self.decode_ar = decode_ar
        self.max_label_length = max_text_length
        self.refine_iters = refine_iters
        self.encoder = encoder
        self.decoder = decoder
        self.text_embed = text_embed
        self.pos_queries = pos_queries
        self.dropout_layer = dropout_layer
        self.dense_head = dense_head

    def decode(
        self,
        tgt,  # shape (B, L)
        memory,  # shape (B, M, d_model)
        tgt_mask=None,
        tgt_padding_mask=None,
        tgt_query=None,
        tgt_query_mask=None,
        training=False,
    ):
        B, L = ops.shape(tgt)[0], ops.shape(tgt)[1]

        # Null context for the bos token
        null_ctx = self.text_embed(tgt[:, :1])  # (B, 1, d_model)

        # If L > 1, embed the rest
        if L > 1:
            tgt_emb_rest = self.text_embed(tgt[:, 1:])  # (B, L-1, d_model)
            # broadcast along batch
            pos_part = self.pos_queries[:, : L - 1]  # shape (1, L-1, d_model)
            tgt_emb = ops.concatenate(
                [null_ctx, tgt_emb_rest + pos_part], axis=1
            )
        else:
            # Only bos token
            tgt_emb = null_ctx

        tgt_emb = self.dropout_layer(tgt_emb, training=training)

        # If no explicit query embeddings are provided, we let
        # the decoder’s "query" = pos_queries up to L
        if tgt_query is None:
            # shape (1, L, d_model) -> tile for batch
            tgt_query = ops.tile(self.pos_queries[:, :L], [B, 1, 1])

        tgt_query = self.dropout_layer(tgt_query, training=training)

        # Pass to the decoder
        out = self.decoder(
            query=tgt_query,
            content=tgt_emb,
            memory=memory,
            query_mask=tgt_query_mask,
            content_mask=tgt_mask,
            content_key_padding_mask=tgt_padding_mask,
            training=training,
        )
        return out

    def forward_test(self, memory, max_length=None):
        testing = max_length is None  # TODO find out what `testing` is for
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = ops.shape(memory)[0]
        num_steps = max_length + 1

        # Prepare positional queries
        pos_queries = ops.tile(self.pos_queries[:, :num_steps], [bs, 1, 1])

        # Create upper triangular mask for autoregressive property
        tgt_mask = query_mask = 1.0 - ops.triu(
            ops.ones([num_steps, num_steps]), 1
        )

        if self.decode_ar:
            tgt_in = ops.full([bs, 1], self.bos_id)
            logits = ops.zeros((bs, 0, self.out_channels - 2), dtype="float32")

            def decode_ar_cond(i, tgt_in, logits):
                return ops.logical_and(
                    i < num_steps,
                    ops.logical_or(
                        not testing,
                        ops.logical_not(
                            ops.all(ops.any(tgt_in == self.eos_id, axis=1))
                        ),
                    ),
                )

            def decode_ar_body(i, tgt_in, logits):
                tgt_out = self.decode(
                    tgt_in[:, : i + 1],
                    memory,
                    tgt_mask[: i + 1, : i + 1],
                    tgt_query=pos_queries[:, i : i + 1],
                    tgt_query_mask=query_mask[i : i + 1, : i + 1],
                )
                p_i = self.dense_head(tgt_out)
                logits = ops.concatenate([logits, p_i], axis=1)
                next_token = ops.argmax(p_i[:, -1, :], axis=-1)
                next_token = ops.expand_dims(next_token, axis=1)
                tgt_in = ops.concatenate([tgt_in, next_token], axis=1)
                return i + 1, tgt_in, logits

            _, tgt_in, logits = ops.while_loop(
                decode_ar_cond, decode_ar_body, [0, tgt_in, logits]
            )
        else:
            tgt_in = ops.full([bs, 1], self.bos_id)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.dense_head(tgt_out)

        if self.refine_iters:
            # Create an upper triangular mask starting from diagonal=2
            query_mask = query_mask * (
                1.0
                - ops.triu(ops.ones([num_steps, num_steps], dtype="float32"), 2)
            )

            bos_tokens = ops.full([bs, 1], self.bos_id)

            def refine_body(i, tgt_in, logits):
                # Regenerate tgt_in based on previous logits
                predictions = ops.argmax(logits[:, :-1, :], axis=-1)
                tgt_in = ops.concatenate([bos_tokens, predictions], axis=1)

                # Create a padding mask for the sequence
                tgt_padding_mask = ops.cast(tgt_in == self.eos_id, "int32")
                tgt_padding_mask = ops.cumsum(tgt_padding_mask, axis=-1) > 0
                tgt_padding_mask = ops.cast(tgt_padding_mask, "float32")

                # Decode and update logits
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    pos_queries,
                    query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.dense_head(tgt_out)
                return [i + 1, tgt_in, logits]

            _, tgt_in, logits = ops.while_loop(
                lambda i, tgt_in, logits: ops.less(i, self.refine_iters),
                refine_body,
                [0, tgt_in, logits],
            )

        # Convert logits to probabilities
        probabilities = keras.activations.softmax(logits, axis=-1)
        return probabilities

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation
        (includes pos. for BOS and EOS tokens)"""
        sz = ops.shape(perm)[0]
        mask = ops.ones((sz, sz))

        for i in range(sz - 1):
            masked_keys = perm[i + 1 : sz]
            query_idx = ops.broadcast_to(perm[i], masked_keys)
            indices = ops.stack(query_idx, masked_keys, axis=1)
            mask = keras.ops.scatter_update(
                mask, indices, keras.ops.zeros(len(masked_keys))
            )

        content_mask = mask[:-1, :-1]
        mask = mask * (1 - ops.eye(sz))
        query_mask = mask[1:, :-1]

        return content_mask, query_mask

    def forward_train(self, memory, tgt, tgt_perms):
        tgt_in = tgt[:, :-1]
        tgt_padding_mask = ops.logical_or(
            tgt_in == self.pad_id, tgt_in == self.eos_id
        )

        logits_list = []

        for perm in tgt_perms:
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.head(
                tgt_in, memory, tgt_mask, tgt_padding_mask, query_mask
            )
            logits = self.dense_head(out)
            logits = ops.reshape(logits, [-1])
            logits_list.append(logits)
        return logits_list

    def call(
        self,
        images,
        training_tgts=None,
        training_tgt_perms=None,
        training=False,
    ):
        memory = self.encoder(images)
        if (
            training
            and training_tgts is not None
            and training_tgt_perms is not None
        ):
            return self.forward_train(memory, training_tgts, training_tgt_perms)
        else:
            return self.forward_test(memory)
