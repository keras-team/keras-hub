import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.t5.t5_backbone import T5Backbone


@keras_hub_export("keras_hub.models.BLIP2FlanT5")
class BLIP2FlanT5(keras.Model):
    """Flan-T5 language model adapter for BLIP-2.

    Wraps ``T5Backbone`` to accept BLIP-2's input format as a functional
    ``keras.Model``.  Inputs:

    - ``token_ids`` / ``padding_mask`` feed the T5 **encoder**.
    - ``qformer_features`` (required when ``num_query_tokens > 0``) are
      projected from ``qformer_hidden_dim`` → ``hidden_dim`` and prepended
      to the encoder token embeddings as a visual soft-prompt.
    - ``decoder_token_ids`` / ``decoder_padding_mask`` feed the T5
      **decoder** (teacher-forced during training, decoder-start + generated
      tokens during inference).

    The model returns the decoder hidden states.  The ``lm_head`` is kept
    as a separate sub-layer so ``BLIP2Seq2SeqLM`` can call it externally.

    Args:
        vocabulary_size: int. Token vocabulary size.
        num_layers: int. Number of T5 encoder and decoder layers each.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Transformer hidden dimension (``d_model``).
        intermediate_dim: int. FFN intermediate dimension (``d_ff``).
        num_query_tokens: int. Number of Q-Former visual query tokens
            prepended to the T5 encoder.  Pass ``0`` for text-only mode.
        qformer_hidden_dim: int. Q-Former output dimension.
        key_value_dim: int or None. Per-head key/value dimension.
            Defaults to ``hidden_dim // num_heads``.
        dropout: float. Dropout probability.
        layer_norm_epsilon: float. Epsilon for T5LayerNorm.
        language_projection: ``keras.layers.Layer`` or None.  Projects
            Q-Former features from ``qformer_hidden_dim`` → ``hidden_dim``.
            A ``Dense`` layer is created when ``None``.
        lm_head: ``keras.layers.Layer`` or None.  Projects decoder hidden
            states → ``vocabulary_size``.  A ``Dense`` layer is created
            when ``None``.
        dtype: dtype for model weights and compute.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_query_tokens,
        qformer_hidden_dim,
        key_value_dim=None,
        dropout=0.1,
        layer_norm_epsilon=1e-6,
        language_projection=None,
        lm_head=None,
        dtype=None,
        **kwargs,
    ):
        t5 = T5Backbone(
            vocabulary_size=vocabulary_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            key_value_dim=key_value_dim,
            dropout=dropout,
            activation="gelu",
            use_gated_activation=True,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="t5",
        )

        if language_projection is None:
            language_projection = keras.layers.Dense(
                hidden_dim,
                use_bias=True,
                dtype=dtype,
                name="language_projection",
            )

        if lm_head is None:
            lm_head = keras.layers.Dense(
                vocabulary_size,
                use_bias=False,
                dtype=dtype,
                name="lm_head",
            )

        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        decoder_token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
            "decoder_token_ids": decoder_token_ids_input,
            "decoder_padding_mask": decoder_padding_mask_input,
        }

        qformer_features_input = None
        if num_query_tokens > 0:
            qformer_features_input = keras.Input(
                shape=(num_query_tokens, qformer_hidden_dim),
                name="qformer_features",
            )
            inputs["qformer_features"] = qformer_features_input

        encoder_out, encoder_attention_mask = self._run_encoder(
            t5,
            language_projection,
            token_ids_input,
            padding_mask_input,
            qformer_features_input,
        )
        decoder_out = self._run_decoder(
            t5,
            decoder_token_ids_input,
            decoder_padding_mask_input,
            encoder_out,
            encoder_attention_mask,
        )

        super().__init__(
            inputs=inputs, outputs=decoder_out, dtype=dtype, **kwargs
        )

        self.t5 = t5
        self.language_projection = language_projection
        self.lm_head = lm_head

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_tokens = num_query_tokens
        self.qformer_hidden_dim = qformer_hidden_dim
        self.key_value_dim = key_value_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    @staticmethod
    def _run_encoder(
        t5, language_projection, token_ids, padding_mask, qformer_features=None
    ):
        enc_emb = t5.token_embedding(token_ids)
        if qformer_features is not None:
            proj = language_projection(qformer_features)
            enc_emb = ops.concatenate([proj, enc_emb], axis=1)
            vis_mask = ops.cast(
                ops.ones_like(qformer_features[..., 0]),
                dtype=padding_mask.dtype,
            )
            full_enc_mask = ops.concatenate([vis_mask, padding_mask], axis=1)
        else:
            full_enc_mask = padding_mask

        x = t5.encoder_embedding_dropout(enc_emb)
        enc_attn_mask = full_enc_mask[:, None, :]
        position_bias = None
        for layer in t5.encoder_transformer_layers:
            out = layer(
                x,
                attention_mask=enc_attn_mask,
                position_bias=position_bias,
                use_causal_mask=False,
            )
            if isinstance(out, tuple):
                x, position_bias = out
        x = t5.encoder_layer_norm(x)
        x = t5.encoder_dropout(x)
        return x, enc_attn_mask

    @staticmethod
    def _run_decoder(
        t5,
        decoder_token_ids,
        decoder_padding_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        dec_emb = t5.token_embedding(decoder_token_ids)
        x = t5.decoder_embedding_dropout(dec_emb)
        dec_attn_mask = decoder_padding_mask[:, None, :]
        position_bias = None
        for layer in t5.decoder_transformer_layers:
            out = layer(
                x,
                attention_mask=dec_attn_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_causal_mask=True,
            )
            if isinstance(out, tuple):
                x, position_bias = out
        x = t5.decoder_layer_norm(x)
        x = t5.decoder_dropout(x)
        return x

    def call_encoder(self, token_ids, padding_mask, qformer_features=None):
        return self._run_encoder(
            self.t5,
            self.language_projection,
            token_ids,
            padding_mask,
            qformer_features,
        )

    def call_decoder(
        self,
        decoder_token_ids,
        decoder_padding_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        return self._run_decoder(
            self.t5,
            decoder_token_ids,
            decoder_padding_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    @property
    def token_embedding(self):
        return self.t5.token_embedding

    @property
    def encoder_transformer_layers(self):
        return self.t5.encoder_transformer_layers

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_tokens": self.num_query_tokens,
                "qformer_hidden_dim": self.qformer_hidden_dim,
                "key_value_dim": self.key_value_dim,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "language_projection": keras.layers.serialize(
                    self.language_projection
                ),
                "lm_head": keras.layers.serialize(self.lm_head),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for key in ("language_projection", "lm_head"):
            if config.get(key) is not None:
                config[key] = keras.layers.deserialize(config[key])
        return cls(**config)
