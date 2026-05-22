"""BLIP-2 Flan-T5 language model."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.t5.t5_backbone import T5Backbone


@keras_hub_export("keras_hub.models.BLIP2FlanT5")
@keras.saving.register_keras_serializable(package="keras_hub")
class BLIP2FlanT5(keras.layers.Layer):
    """Flan-T5 language model adapter for BLIP-2.

    Wraps ``T5Backbone`` to accept BLIP-2's input format:

    - ``token_ids`` / ``padding_mask`` feed the T5 **encoder**.
    - ``qformer_features`` (optional) are projected from
      ``qformer_hidden_dim`` → ``hidden_dim`` and prepended to the
      encoder token embeddings as a visual soft-prompt.
    - ``decoder_token_ids`` / ``decoder_padding_mask`` feed the T5
      **decoder** (teacher-forced during training).  Both default to
      ``token_ids`` / ``padding_mask`` when omitted, which lets
      ``BLIP2Backbone`` trace its functional graph without extra inputs.

    The layer returns the decoder hidden states as a single tensor,
    matching the interface expected by ``BLIP2Backbone`` and
    ``BLIP2CausalLM``.

    Args:
        vocabulary_size: int. Token vocabulary size.
        num_layers: int. Number of T5 encoder and decoder layers each.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Transformer hidden dimension (``d_model``).
        intermediate_dim: int. FFN intermediate dimension (``d_ff``).
        num_query_tokens: int. Number of Q-Former visual query tokens
            prepended to the T5 encoder.
        qformer_hidden_dim: int. Q-Former output dimension (before the
            language projection).
        key_value_dim: int or None. Per-head key/value dimension
            (``d_kv``).  Defaults to ``hidden_dim // num_heads``.
        dropout: float. Dropout probability.
        layer_norm_epsilon: float. Epsilon for T5LayerNorm.
        language_projection: ``keras.layers.Layer`` or None.  Projects
            Q-Former features from ``qformer_hidden_dim`` →
            ``hidden_dim``.  A ``Dense`` layer is created when ``None``.
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
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.t5 = T5Backbone(
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
        self.language_projection = language_projection

        # === Config ===
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

    @property
    def token_embedding(self):
        return self.t5.token_embedding

    def call(self, inputs, training=False):
        enc_ids = inputs["token_ids"]
        enc_mask = inputs["padding_mask"]
        dec_ids = inputs.get("decoder_token_ids", enc_ids)
        dec_mask = inputs.get("decoder_padding_mask", enc_mask)
        qf = inputs.get("qformer_features", None)

        # Encoder
        enc_emb = self.t5.token_embedding(enc_ids)
        if qf is not None:
            proj = self.language_projection(qf)
            enc_emb = ops.concatenate([proj, enc_emb], axis=1)
            vis_mask = ops.cast(ops.ones_like(qf[..., 0]), dtype=enc_mask.dtype)
            enc_mask = ops.concatenate([vis_mask, enc_mask], axis=1)

        x = self.t5.encoder_embedding_dropout(enc_emb, training=training)
        enc_attn_mask = enc_mask[:, None, :]
        position_bias = None
        for layer in self.t5.encoder_transformer_layers:
            out = layer(
                x,
                attention_mask=enc_attn_mask,
                position_bias=position_bias,
                use_causal_mask=False,
                training=training,
            )
            if isinstance(out, tuple):
                x, position_bias = out
        x = self.t5.encoder_layer_norm(x)
        x = self.t5.encoder_dropout(x, training=training)
        encoder_out = x

        # Decoder
        dec_emb = self.t5.token_embedding(dec_ids)
        x = self.t5.decoder_embedding_dropout(dec_emb, training=training)
        dec_attn_mask = dec_mask[:, None, :]
        position_bias = None
        for layer in self.t5.decoder_transformer_layers:
            out = layer(
                x,
                attention_mask=dec_attn_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=enc_attn_mask,
                use_causal_mask=True,
                training=training,
            )
            if isinstance(out, tuple):
                x, position_bias = out
        x = self.t5.decoder_layer_norm(x)
        x = self.t5.decoder_dropout(x, training=training)
        # HF T5ForConditionalGeneration scales decoder output by d_model**-0.5
        # before unembedding when tie_word_embeddings=True (which is always the
        # case for Flan-T5). Apply here so token_embedding(x, reverse=True)
        # gives correct logits without callers needing to know this detail.
        x = x * (self.hidden_dim**-0.5)
        return x

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
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("language_projection") is not None:
            config["language_projection"] = keras.layers.deserialize(
                config["language_projection"]
            )
        return cls(**config)
