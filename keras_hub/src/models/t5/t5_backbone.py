import keras
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.t5.t5_layer_norm import T5LayerNorm
from keras_hub.src.models.t5.t5_transformer_layer import T5TransformerLayer


@keras_hub_export("keras_hub.models.T5Backbone")
class T5Backbone(Backbone):
    """T5 encoder-decoder backbone model.

    T5 is a LLM pretrained on a mix of unsupervised and supervised tasks,
    where each task is converted to a sequence-to-sequence format.
    T5 works well on a variety of tasks out-of-the-box by prepending
    various prefixex to the input sequence, e.g., for translation:
    `"translate English to German: ..."`, for summarization:
    `"summarize: ..."`.

    T5 was introduced in
    [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

    The default constructor gives a fully customizable, randomly initialized T5
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of Transformer layers.
        num_heads: int. The number of attention heads for each Transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The hidden size of the Transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each Transformer layer.
        key_value_dim: int. The dimension of each head of the key/value
            projections in the multi-head attention layers. Defaults to
            hidden_dim / num_heads.
        dropout: float. Dropout probability for the Transformer layers.
        activation: string. The activation function to use in the dense blocks
            of the Transformer Layers.
        use_gated_activation: boolean. Whether to use activation gating in
            the inner dense blocks of the Transformer layers. When used with
            the GELU activation function, this is referred to as GEGLU
            (gated GLU) from https://arxiv.org/pdf/2002.05202.
            The original T5 architecture didn't use gating, but more
            recent versions do. Defaults to `True`.
        layer_norm_epsilon: float. Epsilon factor to be used in the
            layer normalization layers in the Transformer layers.
        tie_embedding_weights: boolean. If `True`, the weights of the token
            embedding and the weights projecting language model outputs from
            `hidden_dim`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        key_value_dim=None,
        dropout=0.1,
        activation="relu",
        use_gated_activation=True,
        layer_norm_epsilon=1e-06,
        tie_embedding_weights=True,
        dtype=None,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.key_value_dim = key_value_dim
        self.dropout = dropout
        self.use_gated_activation = use_gated_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_embedding_weights = tie_embedding_weights

        # Token embedding layer. Shared by encoder and decoder.
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_embedding_weights,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            dtype=dtype,
            name="token_embedding",
        )

        self.encoder_embedding_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="encoder_embedding_dropout"
        )

        self.encoder_transformer_layers = []
        for i in range(num_layers):
            self.encoder_transformer_layers.append(
                T5TransformerLayer(
                    is_decoder=False,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    key_value_dim=key_value_dim or hidden_dim // num_heads,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_epsilon=layer_norm_epsilon,
                    num_heads=num_heads,
                    use_gated_activation=use_gated_activation,
                    use_relative_attention_bias=bool(i == 0),
                    dtype=dtype,
                    name=f"transformer_encoder_layer_{i}",
                )
            )

        self.encoder_layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="encoder_output_layer_norm",
        )
        self.encoder_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="encoder_output_dropout"
        )

        self.decoder_embedding_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="decoder_embedding_dropout"
        )

        self.decoder_transformer_layers = []
        for i in range(num_layers):
            self.decoder_transformer_layers.append(
                T5TransformerLayer(
                    is_decoder=True,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    key_value_dim=key_value_dim or hidden_dim // num_heads,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_epsilon=layer_norm_epsilon,
                    num_heads=num_heads,
                    use_gated_activation=use_gated_activation,
                    use_relative_attention_bias=bool(i == 0),
                    dtype=dtype,
                    name=f"transformer_decoder_layer_{i}",
                )
            )

        self.decoder_layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="decoder_output_layer_norm",
        )
        self.decoder_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="decoder_output_dropout"
        )

        # NNX Initialization
        nnx_enabled = keras.config.is_nnx_enabled()
        if not nnx_enabled:
            # === Functional Model ===
            encoder_token_id_input = keras.Input(
                shape=(None,), dtype="int32", name="encoder_token_ids"
            )
            encoder_padding_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="encoder_padding_mask"
            )
            decoder_token_id_input = keras.Input(
                shape=(None,), dtype="int32", name="decoder_token_ids"
            )
            decoder_padding_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="decoder_padding_mask"
            )

            outputs = self._forward(
                {
                    "encoder_token_ids": encoder_token_id_input,
                    "encoder_padding_mask": encoder_padding_mask_input,
                    "decoder_token_ids": decoder_token_id_input,
                    "decoder_padding_mask": decoder_padding_mask_input,
                },
                training=None,
            )

            super().__init__(
                inputs={
                    "encoder_token_ids": encoder_token_id_input,
                    "encoder_padding_mask": encoder_padding_mask_input,
                    "decoder_token_ids": decoder_token_id_input,
                    "decoder_padding_mask": decoder_padding_mask_input,
                },
                outputs=outputs,
                dtype=dtype,
                **kwargs,
            )
        else:
            super().__init__(dtype=dtype, **kwargs)

    def call(self, inputs, training=None):
        return self._forward(inputs, training=training)

    # Config
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "activation": keras.activations.serialize(self.activation),
                "key_value_dim": self.key_value_dim,
                "dropout": self.dropout,
                "use_gated_activation": self.use_gated_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "tie_embedding_weights": self.tie_embedding_weights,
            }
        )
        return config

    def _forward(self, inputs, training=None):
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]

        # Encoder
        x = self.token_embedding(encoder_token_ids)
        x = self.encoder_embedding_dropout(x, training=training)

        encoder_attention_mask = encoder_padding_mask[:, None, :]
        position_bias = None

        for transformer_layer in self.encoder_transformer_layers:
            output = transformer_layer(
                x,
                attention_mask=encoder_attention_mask,
                position_bias=position_bias,
                use_causal_mask=False,
                training=training,
            )
            if isinstance(output, tuple):
                x, position_bias = output

        encoder_output = self.encoder_dropout(
            self.encoder_layer_norm(x), training=training
        )

        # Decoder
        x = self.token_embedding(decoder_token_ids)
        x = self.decoder_embedding_dropout(x, training=training)

        decoder_attention_mask = decoder_padding_mask[:, None, :]
        position_bias = None

        for transformer_layer in self.decoder_transformer_layers:
            output = transformer_layer(
                x,
                attention_mask=decoder_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
                use_causal_mask=True,
                training=training,
            )
            if isinstance(output, tuple):
                x, position_bias = output

        decoder_output = self.decoder_dropout(
            self.decoder_layer_norm(x), training=training
        )

        return {
            "encoder_sequence_output": encoder_output,
            "decoder_sequence_output": decoder_output,
        }
