import keras
import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.parseq.parseq_head import Decoder
from keras_hub.src.models.parseq.parseq_vit import VisionTransformer


@keras_hub_export("keras_hub.models.PARSeqBackbone")
class PARSeqBackbone(Backbone):
    """Scene Text Detection with PARSeq.

    Performs OCR in natural scenes using the PARSeq model described in [Scene
    Text Recognition with Permuted Autoregressive Sequence Models](
    https://arxiv.org/abs/2207.06966). PARSeq is a ViT-based model that allows
    iterative decoding by performing an autoregressive decoding phase, followed
    by a refinement phase.

    Args:
        decode_autoregressive: bool. Whether to perform an autoregressive
            decoding phase. Defaults to True.
        refine_iterations: int. The number of iterations for refining the
            prediction. Defaults to 1.
        alphabet_size: int. The number of possible output characters.
            Defaults to 97.
        max_text_length: int. The maximum output text length. Defaults to 25.
        patch_size: tuple of ints. The patch size used by the Vision
            Transformer. Defaults to (4, 8).
        embed_dim: int. The dimensionality of the used embedding vectors.
            Defaults to 384.
        mlp_dim: int. The dimensionality of intermediate MLP layers in
            each transformer layer. Defaults to 1536.
        enc_depth: int. The number of encoder layers. Defaults to 12.
        num_enc_heads: int. The number of encoder attention heads.
            Defaults to 6.
        dec_depth: int. The number of decoder layers. Defaults to 1.
        num_dec_heads: int. The number of decoder attention heads.
            Defaults to 12.
        dropout_rate: int. The dropout rate for embedding vectors.
            Defaults to 0.1.
    """

    def __init__(
        self,
        decode_autoregressive=True,
        refine_iterations=1,
        alphabet_size=97,
        max_text_length=25,
        patch_size=(4, 8),
        embed_dim=384,
        mlp_dim=1536,
        enc_depth=12,
        num_enc_heads=6,
        dec_depth=1,
        num_dec_heads=12,
        dropout_rate=0.1,
        **kwargs,
    ):
        encoder = VisionTransformer(
            patch_size=patch_size,
            class_num=None,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            depth=enc_depth,
            num_heads=num_enc_heads,
            name="parseq_vit",
        )
        decoder = Decoder(
            num_layers=dec_depth,
            embed_dim=embed_dim,
            num_heads=num_dec_heads,
            mlp_dim=mlp_dim,
            dropout=dropout_rate,
            activation="gelu",
            name="parseq_dec",
        )
        # Embedding layer for input tokens
        text_embed = layers.Embedding(
            input_dim=alphabet_size, output_dim=embed_dim, name="parseq_embed"
        )
        dropout = layers.Dropout(dropout_rate, name="parseq_dropout")
        # Output head to project decoder outputs to token probabilities
        dense_head = layers.Dense(alphabet_size - 2, name="parseq_head")

        super().__init__(**kwargs)

        pos_query_embeddings = self.add_weight(
            name="parseq_pos_query_embeddings",
            shape=(1, max_text_length + 1, embed_dim),
            initializer="zeros",
            trainable=True,
        )

        self.decode_autoregressive = decode_autoregressive
        self.refine_iterations = refine_iterations
        self.alphabet_size = alphabet_size
        self.max_text_length = max_text_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.enc_depth = enc_depth
        self.num_enc_heads = num_enc_heads
        self.dec_depth = dec_depth
        self.num_dec_heads = num_dec_heads
        self.dropout_rate = dropout_rate

        self.encoder = encoder
        self.decoder = decoder
        self.text_embed = text_embed
        self.pos_query_embeddings = pos_query_embeddings
        self.dropout = dropout
        self.dense_head = dense_head

    @property
    def bos_id(self):
        return self.alphabet_size - 2

    @property
    def eos_id(self):
        return 0

    @property
    def pad_id(self):
        return self.alphabet_size - 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decode_autoregressive": self.decode_autoregressive,
                "refine_iterations": self.refine_iterations,
                "alphabet_size": self.alphabet_size,
                "max_text_length": self.max_text_length,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "enc_depth": self.enc_depth,
                "num_enc_heads": self.num_enc_heads,
                "dec_depth": self.dec_depth,
                "num_dec_heads": self.num_dec_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def decode(
        self,
        tokens_in,
        memory,
        token_mask=None,
        token_padding_mask=None,
        token_query=None,
        token_query_mask=None,
        training=False,
    ):
        batch_size, input_length = ops.shape(tokens_in)[:2]

        # Null context for the BOS token
        null_ctx = self.embed_dim**0.5 * self.text_embed(tokens_in[:, :1])
        # Token embeddings & positional embeddings for remaining tokens
        token_emb = self.embed_dim**0.5 * self.text_embed(tokens_in[:, 1:])
        pos_embeddings = self.pos_query_embeddings[:, : input_length - 1]
        token_emb = ops.concatenate(
            [null_ctx, token_emb + pos_embeddings], axis=1
        )
        token_emb = self.dropout(token_emb, training=training)

        # If no explicit query embeddings are provided, we let
        # the decoder’s "query" = pos_query_embeddings up to `input_length`
        if token_query is None:
            token_query = (
                ops.ones((batch_size, 1, 1))
                * self.pos_query_embeddings[:, :input_length]
            )
        token_query = self.dropout(token_query, training=training)
        out = self.decoder(
            query=token_query,
            content=token_emb,
            memory=memory,
            query_mask=token_query_mask,
            content_mask=token_mask,
            content_key_padding_mask=token_padding_mask,
            training=training,
        )
        return out

    def forward_test(self, memory, max_length=None):
        batch_size = ops.shape(memory)[0]
        max_length = (
            self.max_text_length
            if max_length is None
            else min(max_length, self.max_text_length)
        )
        num_steps = max_length + 1

        # For autoregressive decoding, the number of tokens that we feed into
        # the model increases with each iteration. Since Jax does not support
        # dynamically-sized tensors, we have to build a graph with unrolled AR
        # loop iterations when working with Jax
        unroll_ar_loop = keras.config.backend() == "jax"

        # Prepare positional queries
        pos_query_embeddings = (
            ops.ones((batch_size, 1, 1))
            * self.pos_query_embeddings[:, :num_steps]
        )
        # Create upper triangular mask for autoregressive property
        token_mask = query_mask = ops.convert_to_tensor(
            1.0 - np.triu(np.ones([num_steps, num_steps]), 1)
        )

        if self.decode_autoregressive:
            tokens_in = ops.concatenate(
                (
                    ops.full([batch_size, 1], self.bos_id),
                    ops.full([batch_size, num_steps - 1], self.pad_id),
                ),
                axis=1,
            )
            tokens_in = ops.cast(tokens_in, "int32")
            logits = ops.zeros(
                (batch_size, num_steps, self.alphabet_size - 2), "float32"
            )

            def decode_ar_cond(i, tokens_in, logits):
                return ops.logical_and(
                    i < num_steps,
                    ops.logical_not(
                        ops.all(ops.any(tokens_in == self.eos_id, axis=1))
                    ),
                )

            def decode_ar_body(i, tokens_in, logits):
                token_out = self.decode(
                    tokens_in[:, : i + 1],
                    memory,
                    token_mask[: i + 1, : i + 1],
                    token_query=pos_query_embeddings[:, i : i + 1],
                    token_query_mask=query_mask[i : i + 1, : i + 1],
                )
                p_i = self.dense_head(token_out)
                logits = ops.slice_update(logits, (0, i, 0), p_i)
                next_token = ops.argmax(p_i[:, -1, :], axis=-1)
                next_token = ops.expand_dims(next_token, axis=1)
                tokens = ops.slice_update(tokens_in, (0, i + 1), next_token)
                return i + 1, tokens, logits

            if unroll_ar_loop:
                for i in range(num_steps):
                    _, tokens_in, logits = ops.cond(
                        decode_ar_cond(i, tokens_in, logits),
                        lambda: decode_ar_body(i, tokens_in, logits),
                        lambda: (i, tokens_in, logits),
                    )
            else:
                _, tokens_in, logits = ops.while_loop(
                    decode_ar_cond, decode_ar_body, [0, tokens_in, logits]
                )
        else:
            tokens_in = ops.full([batch_size, 1], self.bos_id, "int32")
            tokens_out = self.decode(
                tokens_in, memory, token_query=pos_query_embeddings
            )
            logits = self.dense_head(tokens_out)

        if self.refine_iterations:
            # Create an upper triangular mask starting from diagonal=2
            query_mask = query_mask * ops.convert_to_tensor(
                1.0 - np.triu(np.ones([num_steps, num_steps]), 2)
            )
            bos_tokens = ops.full([batch_size, 1], self.bos_id, "int32")

            def refine_body(i, logits):
                # Regenerate tokens_in based on previous logits
                predictions = ops.argmax(logits[:, :-1, :], axis=-1)
                tokens_in = ops.concatenate([bos_tokens, predictions], axis=1)
                # Create a padding mask for the sequence
                token_padding_mask = ops.cast(tokens_in == self.eos_id, "int32")
                token_padding_mask = ops.cumsum(token_padding_mask, axis=-1) > 0
                token_padding_mask = ops.cast(token_padding_mask, "float32")
                # Decode and update logits
                tokens_out = self.decode(
                    tokens_in,
                    memory,
                    token_mask,
                    token_padding_mask,
                    pos_query_embeddings,
                    query_mask[:, : tokens_in.shape[1]],
                )
                logits = self.dense_head(tokens_out)
                return logits

            logits = ops.fori_loop(
                0, self.refine_iterations, refine_body, logits
            )

        # Convert logits to probabilities
        probabilities = keras.activations.softmax(logits, axis=-1)
        return probabilities

    def generate_attention_masks(self, perm):
        """Generate attention masks given a sequence permutation
        (includes pos. for BOS and EOS tokens)"""
        input_length = perm.shape[0]
        mask = ops.ones((input_length, input_length))
        for i in range(input_length - 1):
            masked_keys = perm[i + 1 : input_length]
            query_idx = ops.broadcast_to(perm[i], ops.shape(masked_keys))
            indices = ops.stack((query_idx, masked_keys), axis=1)
            mask = keras.ops.scatter_update(
                mask, indices, keras.ops.zeros(ops.shape(masked_keys)[0])
            )
        content_mask = mask[:-1, :-1]
        mask = mask * (1 - ops.eye(input_length))
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def forward_train(self, memory, tokens_in, token_perms):
        token_padding_mask = ops.logical_or(
            tokens_in == self.pad_id, tokens_in == self.eos_id
        )
        token_padding_mask = ops.cast(token_padding_mask, "float32")
        logits_list = []
        for i in range(token_perms.shape[0]):
            token_mask, query_mask = self.generate_attention_masks(
                token_perms[i]
            )
            out = self.decode(
                tokens_in=tokens_in,
                memory=memory,
                token_mask=token_mask,
                token_padding_mask=token_padding_mask,
                token_query_mask=query_mask,
            )
            logits = self.dense_head(out)
            logits_list.append(logits)
        return ops.stack(logits_list, axis=0)

    def call(
        self,
        images,
        training_tokens=None,
        training_token_perms=None,
        training=False,
    ):
        memory = self.encoder(images)
        if (
            training
            and training_tokens is not None
            and training_token_perms is not None
        ):
            return self.forward_train(
                memory, training_tokens, training_token_perms
            )
        else:
            return self.forward_test(memory)

    def build(self, input_shape):
        # trigger building of the layers by running some dummy input through
        # them. this is mainly needed for Jax to properly initialize the
        # layers' weights
        batch_size = input_shape[0] or 1
        memory = self.encoder(
            ops.zeros((batch_size, *input_shape[1:]), "float32")
        )
        token_out = self.decode(ops.zeros((batch_size, 1), "int32"), memory)
        self.dense_head(token_out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_text_length, self.alphabet_size)
