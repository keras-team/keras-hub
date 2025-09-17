import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.t5gemma.t5gemma_decoder import T5GemmaDecoderLayer
from keras_hub.src.models.t5gemma.t5gemma_encoder import T5GemmaEncoderLayer
from keras_hub.src.models.t5gemma.t5gemma_layers import (
    t5gemma_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras_hub_export("keras_hub.models.T5GemmaBackbone")
class T5GemmaBackbone(Backbone):
    """T5Gemma backbone model.

    This class implements the encoder-decoder backbone of the T5Gemma model,
    consisting of an embedding layer, a stack of encoder layers, and a
    stack of decoder layers.

    Args:
        vocabulary_size: int, The size of the vocabulary.
        encoder_hidden_dim: int, The hidden dimensionality of the encoder.
        encoder_intermediate_dim: int, The intermediate size of the encoder's
            feed-forward networks.
        encoder_num_layers: int, The number of encoder layers.
        encoder_num_attention_heads: int, The number of attention heads in the
            encoder.
        encoder_num_key_value_heads: int, The number of key-value heads in the
            encoder.
        encoder_head_dim: int, The dimensionality of each attention head in the
            encoder.
        encoder_layer_types: list of str, A list of strings specifying the type
            of attention layer for each encoder layer. Each element can be
            either `"sliding_attention"` or `"full_attention"`. For example,
            `["full_attention", "sliding_attention", ...]`.
        decoder_hidden_dim: int, The hidden dimensionality of the decoder.
        decoder_intermediate_dim: int, The intermediate size of the decoder's
            feed-forward networks.
        decoder_num_layers: int, The number of decoder layers.
        decoder_num_attention_heads: int, The number of attention heads in the
            decoder.
        decoder_num_key_value_heads: int, The number of key-value heads in the
            decoder.
        decoder_head_dim: int, The dimensionality of each attention head in the
            decoder.
        decoder_layer_types: list of str, A list of strings specifying the type
            of attention layer for each decoder layer. Each element can be
            either `"sliding_attention"` or `"full_attention"`. For example,
            `["full_attention", "sliding_attention", ...]`.
        dropout_rate: float, The dropout rate applied throughout the model.
            Defaults to `0.0`.
        rms_norm_eps: float, The epsilon value for RMS normalization. Defaults
            to `1e-6`.
        query_pre_attn_scalar: float, Scalar to multiply queries by before
            attention. Defaults to `1.0`.
        attention_bias: bool, Whether to include bias in attention computations.
            Defaults to `False`.
        hidden_activation: str, The activation function used in the feed-forward
            networks. Defaults to `"gelu_approximate"`.
        tie_word_embeddings: bool, Whether to tie input and output word
            embeddings. Defaults to `True`.
        initializer_range: float, The range for the random normal initializer.
            Defaults to `0.02`.
        attention_dropout: float, The dropout rate applied to attention weights.
            Defaults to `0.0`.
        sliding_window: int, optional, The window size for sliding attention.
            Required if any `layer_type` is `"sliding_attention"`. Defaults to
            `None`.
        cross_attention_hidden_size: int, optional, The hidden size for
            cross-attention in the decoder layers. If None, it defaults to
            `encoder_hidden_dim`. Defaults to `None`.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits. Defaults to `None`.
        final_logit_softcapping: float, optional, The softcapping value for
            final logits. Defaults to `None`.
        rope_max_wavelength: float, The maximum wavelength for Rotary Positional
            Embeddings. Defaults to `10000.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent `Backbone`
            class.

    Examples:
    ```python
    import numpy as np
    from keras_hub.models import T5GemmaBackbone

    input_data = {
        "encoder_token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "encoder_padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype="int32"
        ),
        "decoder_token_ids": np.ones(shape=(1, 8), dtype="int32"),
        "decoder_padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1]], dtype="int32"
        ),
    }

    # Randomly initialized T5Gemma backbone with custom config.
    model = T5GemmaBackbone(
        vocabulary_size=32000,
        # Encoder parameters.
        encoder_hidden_dim=256,
        encoder_intermediate_dim=512,
        encoder_num_layers=4,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_head_dim=64,
        encoder_layer_types=["full_attention"] * 4,
        # Decoder parameters.
        decoder_hidden_dim=256,
        decoder_intermediate_dim=512,
        decoder_num_layers=4,
        decoder_num_attention_heads=4,
        decoder_num_key_value_heads=2,
        decoder_head_dim=64,
        decoder_layer_types=["full_attention"] * 4,
        # Common parameters.
        dropout_rate=0.1,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=1.0,
        attention_bias=False,
        hidden_activation="gelu_approximate",
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        encoder_hidden_dim,
        encoder_intermediate_dim,
        encoder_num_layers,
        encoder_num_attention_heads,
        encoder_num_key_value_heads,
        encoder_head_dim,
        encoder_layer_types,
        decoder_hidden_dim,
        decoder_intermediate_dim,
        decoder_num_layers,
        decoder_num_attention_heads,
        decoder_num_key_value_heads,
        decoder_head_dim,
        decoder_layer_types,
        dropout_rate=0.0,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=1.0,
        attention_bias=False,
        hidden_activation="gelu_approximate",
        tie_word_embeddings=True,
        initializer_range=0.02,
        attention_dropout=0.0,
        sliding_window=None,
        cross_attention_hidden_size=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        rope_max_wavelength=10000.0,
        dtype=None,
        **kwargs,
    ):
        self.kernel_initializer = t5gemma_kernel_initializer(initializer_range)

        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=encoder_hidden_dim,
            embeddings_initializer=clone_initializer(self.kernel_initializer),
            dtype=dtype,
            name="encoder_token_embedding",
        )
        self.decoder_token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=decoder_hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=clone_initializer(self.kernel_initializer),
            dtype=dtype,
            name="decoder_token_embedding",
        )
        self.encoder_layers = [
            T5GemmaEncoderLayer(
                hidden_size=encoder_hidden_dim,
                rms_norm_eps=rms_norm_eps,
                num_attention_heads=encoder_num_attention_heads,
                num_key_value_heads=encoder_num_key_value_heads,
                query_pre_attn_scalar=query_pre_attn_scalar,
                attention_bias=attention_bias,
                intermediate_size=encoder_intermediate_dim,
                hidden_activation=hidden_activation,
                head_dim=encoder_head_dim,
                dropout_rate=dropout_rate,
                initializer_range=initializer_range,
                attention_dropout=attention_dropout,
                layer_type=encoder_layer_types[i],
                sliding_window=sliding_window,
                attn_logit_softcapping=attn_logit_softcapping,
                rope_max_wavelength=rope_max_wavelength,
                name=f"encoder_layer_{i}",
                dtype=dtype,
            )
            for i in range(encoder_num_layers)
        ]
        self.encoder_norm = RMSNormalization(epsilon=rms_norm_eps, dtype=dtype)
        self.encoder_dropout = keras.layers.Dropout(dropout_rate, dtype=dtype)
        self.decoder_layers = [
            T5GemmaDecoderLayer(
                hidden_size=decoder_hidden_dim,
                rms_norm_eps=rms_norm_eps,
                num_attention_heads=decoder_num_attention_heads,
                num_key_value_heads=decoder_num_key_value_heads,
                query_pre_attn_scalar=query_pre_attn_scalar,
                attention_bias=attention_bias,
                intermediate_size=decoder_intermediate_dim,
                hidden_activation=hidden_activation,
                dropout_rate=dropout_rate,
                initializer_range=initializer_range,
                head_dim=decoder_head_dim,
                attention_dropout=attention_dropout,
                layer_type=decoder_layer_types[i],
                sliding_window=sliding_window,
                cross_attention_hidden_size=(
                    cross_attention_hidden_size or encoder_hidden_dim
                ),
                attn_logit_softcapping=attn_logit_softcapping,
                rope_max_wavelength=rope_max_wavelength,
                name=f"decoder_layer_{i}",
                dtype=dtype,
            )
            for i in range(decoder_num_layers)
        ]
        self.decoder_norm = RMSNormalization(epsilon=rms_norm_eps, dtype=dtype)
        self.decoder_dropout = keras.layers.Dropout(dropout_rate, dtype=dtype)

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

        # Encoder.
        encoder_embeddings = self.token_embedding(encoder_token_id_input)
        encoder_embeddings = encoder_embeddings * keras.ops.cast(
            keras.ops.sqrt(encoder_hidden_dim), encoder_embeddings.dtype
        )
        encoder_hidden_states = self.encoder_dropout(encoder_embeddings)
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(
                encoder_hidden_states, padding_mask=encoder_padding_mask_input
            )
        encoder_output = self.encoder_norm(encoder_hidden_states)
        encoder_output = self.encoder_dropout(encoder_output)

        # Decoder.
        decoder_embeddings = self.decoder_token_embedding(
            decoder_token_id_input
        )
        decoder_embeddings = decoder_embeddings * keras.ops.cast(
            keras.ops.sqrt(decoder_hidden_dim), decoder_embeddings.dtype
        )
        decoder_hidden_states = self.decoder_dropout(decoder_embeddings)
        for layer in self.decoder_layers:
            decoder_hidden_states, _ = layer(
                (decoder_hidden_states, encoder_output),
                self_attention_padding_mask=decoder_padding_mask_input,
                cross_attention_padding_mask=encoder_padding_mask_input,
            )
        decoder_output = self.decoder_norm(decoder_hidden_states)
        decoder_output = self.decoder_dropout(decoder_output)

        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask_input,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask_input,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_intermediate_dim = encoder_intermediate_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = encoder_num_key_value_heads
        self.encoder_head_dim = encoder_head_dim
        self.encoder_layer_types = encoder_layer_types
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_intermediate_dim = decoder_intermediate_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_num_key_value_heads = decoder_num_key_value_heads
        self.decoder_head_dim = decoder_head_dim
        self.decoder_layer_types = decoder_layer_types
        self.vocabulary_size = vocabulary_size
        self.dropout_rate = dropout_rate
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.cross_attention_hidden_size = (
            cross_attention_hidden_size or encoder_hidden_dim
        )
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.rope_max_wavelength = rope_max_wavelength

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_intermediate_dim": self.encoder_intermediate_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_num_attention_heads": self.encoder_num_attention_heads,
                "encoder_num_key_value_heads": self.encoder_num_key_value_heads,
                "encoder_layer_types": self.encoder_layer_types,
                "encoder_head_dim": self.encoder_head_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "decoder_intermediate_dim": self.decoder_intermediate_dim,
                "decoder_num_layers": self.decoder_num_layers,
                "decoder_num_attention_heads": self.decoder_num_attention_heads,
                "decoder_num_key_value_heads": self.decoder_num_key_value_heads,
                "decoder_layer_types": self.decoder_layer_types,
                "decoder_head_dim": self.decoder_head_dim,
                "dropout_rate": self.dropout_rate,
                "rms_norm_eps": self.rms_norm_eps,
                "tie_word_embeddings": self.tie_word_embeddings,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attention_bias": self.attention_bias,
                "hidden_activation": self.hidden_activation,
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "sliding_window": self.sliding_window,
                "cross_attention_hidden_size": self.cross_attention_hidden_size,
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "final_logit_softcapping": self.final_logit_softcapping,
                "rope_max_wavelength": self.rope_max_wavelength,
            }
        )
        return config
