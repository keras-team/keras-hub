import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma3.gemma3_layers import Gemma3InterleaveEmbeddings
from keras_hub.src.models.gemma3.gemma3_layers import RMSNormalization
from keras_hub.src.models.t5gemma2.t5gemma2_decoder import T5Gemma2DecoderLayer
from keras_hub.src.models.t5gemma2.t5gemma2_encoder import T5Gemma2EncoderLayer
from keras_hub.src.models.t5gemma2.t5gemma2_layers import (
    t5gemma2_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras_hub_export("keras_hub.models.T5Gemma2Backbone")
class T5Gemma2Backbone(Backbone):
    """T5Gemma2 backbone model.

    This class implements the encoder-decoder backbone of the T5Gemma2
    model. T5Gemma2 is based on Gemma3 and features merged
    self+cross attention in the decoder (unlike T5Gemma which used
    separate attention sublayers), Gemma3-style Q/K normalization,
    and per-layer-type sliding window attention patterns.

    When a `vision_encoder` is provided, the model also accepts image
    inputs. Images are processed by the vision encoder and the resulting
    embeddings are interleaved into the encoder text embeddings at
    positions marked by image placeholder tokens.

    Args:
        vocabulary_size: int, The size of the vocabulary.
        encoder_hidden_dim: int, Encoder hidden dimensionality.
        encoder_intermediate_dim: int, Encoder FFN intermediate size.
        encoder_num_layers: int, Number of encoder layers.
        encoder_num_attention_heads: int, Encoder attention heads.
        encoder_num_key_value_heads: int, Encoder KV heads for GQA.
        encoder_head_dim: int, Encoder head dimensionality.
        encoder_layer_types: list of str, Attention layer types for
            each encoder layer (`"full_attention"` or
            `"sliding_attention"`).
        decoder_hidden_dim: int, Decoder hidden dimensionality.
        decoder_intermediate_dim: int, Decoder FFN intermediate size.
        decoder_num_layers: int, Number of decoder layers.
        decoder_num_attention_heads: int, Decoder attention heads.
        decoder_num_key_value_heads: int, Decoder KV heads for GQA.
        decoder_head_dim: int, Decoder head dimensionality.
        decoder_layer_types: list of str, Attention layer types for
            each decoder layer.
        dropout_rate: float, Dropout rate. Defaults to `0.0`.
        rms_norm_eps: float, RMS normalization epsilon. Defaults to
            `1e-6`.
        query_pre_attn_scalar: float, Query scalar. Defaults to `1.0`.
        attention_bias: bool, Attention bias. Defaults to `False`.
        hidden_activation: str, FFN activation. Defaults to
            `"gelu_approximate"`.
        tie_word_embeddings: bool, Tie input/output embeddings.
            Defaults to `True`.
        initializer_range: float, Initializer range. Defaults to
            `0.02`.
        attention_dropout: float, Attention dropout. Defaults to `0.0`.
        sliding_window: int, optional, Sliding window size.
        cross_attention_hidden_size: int, optional, Cross-attention
            hidden size. Defaults to `encoder_hidden_dim`.
        attn_logit_softcapping: float, optional, Attention softcapping.
        final_logit_softcapping: float, optional, Final logit
            softcapping.
        rope_max_wavelength: float, RoPE maximum wavelength.
            Defaults to `10000.0`.
        global_rope_scaling_factor: float, RoPE scaling factor for
            full attention layers. Defaults to `1.0`.
        use_query_key_norm: bool, Whether to use Gemma3-style Q/K
            normalization. Defaults to `True`.
        vision_encoder: optional, A `Gemma3VisionEncoder` instance for
            multimodal inputs. When `None`, the model is text-only.
        eoi_token_index: int, Token index for the end-of-image token.
            Defaults to `256000`.
        dtype: dtype for computations. Defaults to `None`.
        **kwargs: Additional keyword arguments.

    Examples:
    ```python
    import numpy as np
    from keras_hub.models import T5Gemma2Backbone

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

    model = T5Gemma2Backbone(
        vocabulary_size=32000,
        encoder_hidden_dim=256,
        encoder_intermediate_dim=512,
        encoder_num_layers=4,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_head_dim=64,
        encoder_layer_types=["full_attention"] * 4,
        decoder_hidden_dim=256,
        decoder_intermediate_dim=512,
        decoder_num_layers=4,
        decoder_num_attention_heads=4,
        decoder_num_key_value_heads=2,
        decoder_head_dim=64,
        decoder_layer_types=["full_attention"] * 4,
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
        global_rope_scaling_factor=1.0,
        encoder_rope_max_wavelength=None,
        encoder_global_rope_scaling_factor=None,
        use_query_key_norm=True,
        vision_encoder=None,
        eoi_token_index=256000,
        dtype=None,
        **kwargs,
    ):
        self.kernel_initializer = t5gemma2_kernel_initializer(initializer_range)

        # Determine if text-only.
        self.vision_encoder = vision_encoder
        text_only_model = vision_encoder is None

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

        # Vision interleaving layer (only when vision encoder is present).
        if not text_only_model:
            self.interleave_embeddings = Gemma3InterleaveEmbeddings(
                num_vision_tokens_per_image=self.vision_encoder.num_vision_tokens_per_image,
                dtype=dtype,
                name="interleave_embeddings",
            )
            # EOI (end-of-image) embeddings: learned vectors that
            # replace the standard embedding at eoi_token_index.
            self.encoder_eoi_embedding = keras.Variable(
                ops.zeros((encoder_hidden_dim,)),
                name="encoder_eoi_embedding",
            )
            self.decoder_eoi_embedding = keras.Variable(
                ops.zeros((decoder_hidden_dim,)),
                name="decoder_eoi_embedding",
            )

        # Encoder may have different RoPE config than decoder.
        enc_rope = (
            encoder_rope_max_wavelength
            if encoder_rope_max_wavelength is not None
            else rope_max_wavelength
        )
        enc_rope_factor = (
            encoder_global_rope_scaling_factor
            if encoder_global_rope_scaling_factor is not None
            else global_rope_scaling_factor
        )

        self.encoder_layers = []
        for i in range(encoder_num_layers):
            # Per-layer RoPE wavelength: base for sliding, 1M for global.
            layer_rope = (
                enc_rope
                if encoder_layer_types[i] == "sliding_attention"
                else 1_000_000.0
            )
            # Per-layer RoPE scaling: 1.0 for sliding (default),
            # global_rope_scaling_factor for full_attention (linear).
            layer_rope_factor = (
                1.0
                if encoder_layer_types[i] == "sliding_attention"
                else enc_rope_factor
            )
            self.encoder_layers.append(
                T5Gemma2EncoderLayer(
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
                    rope_max_wavelength=layer_rope,
                    rope_scaling_factor=layer_rope_factor,
                    use_query_key_norm=use_query_key_norm,
                    name=f"encoder_layer_{i}",
                    dtype=dtype,
                )
            )
        self.encoder_norm = RMSNormalization(epsilon=rms_norm_eps, dtype=dtype)
        self.encoder_dropout = keras.layers.Dropout(dropout_rate, dtype=dtype)
        self.decoder_layers = []
        for i in range(decoder_num_layers):
            # Per-layer RoPE wavelength: 10K for sliding, 1M for global.
            layer_rope = (
                rope_max_wavelength
                if decoder_layer_types[i] == "sliding_attention"
                else 1_000_000.0
            )
            # Per-layer RoPE scaling: 1.0 for sliding (default),
            # global_rope_scaling_factor for full_attention (linear).
            layer_rope_factor = (
                1.0
                if decoder_layer_types[i] == "sliding_attention"
                else global_rope_scaling_factor
            )
            self.decoder_layers.append(
                T5Gemma2DecoderLayer(
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
                    rope_max_wavelength=layer_rope,
                    rope_scaling_factor=layer_rope_factor,
                    use_query_key_norm=use_query_key_norm,
                    name=f"decoder_layer_{i}",
                    dtype=dtype,
                )
            )
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

        # Optional vision inputs.
        if not text_only_model:
            image_size = self.vision_encoder.image_size
            image_input = keras.Input(
                shape=(None, image_size, image_size, 3),
                name="images",
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )

        # Encoder.
        encoder_embeddings = self.token_embedding(encoder_token_id_input)
        encoder_embeddings = encoder_embeddings * ops.cast(
            ops.sqrt(encoder_hidden_dim), encoder_embeddings.dtype
        )

        # Handle EOI embedding replacement.
        if not text_only_model:
            # Replace embeddings at eoi_token_index positions with the
            # learned eoi_embedding (a separate parameter per HF design).
            # Use ops.where with automatic broadcasting (no broadcast_to
            # needed — avoids issues with symbolic shapes during tracing).
            eoi_mask = ops.cast(
                ops.expand_dims(
                    ops.equal(encoder_token_id_input, eoi_token_index),
                    axis=-1,
                ),
                encoder_embeddings.dtype,
            )
            encoder_embeddings = (
                eoi_mask * self.encoder_eoi_embedding
                + (1 - eoi_mask) * encoder_embeddings
            )

        # Interleave vision embeddings if images are provided.
        if not text_only_model:
            img_embeddings = self.vision_encoder(image_input)
            encoder_embeddings = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=encoder_embeddings,
                vision_indices=vision_indices_input,
            )

        encoder_hidden_states = self.encoder_dropout(encoder_embeddings)
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(
                encoder_hidden_states,
                padding_mask=encoder_padding_mask_input,
            )
        encoder_output = self.encoder_norm(encoder_hidden_states)
        encoder_output = self.encoder_dropout(encoder_output)

        # Decoder.
        decoder_embeddings = self.decoder_token_embedding(
            decoder_token_id_input
        )
        decoder_embeddings = decoder_embeddings * ops.cast(
            ops.sqrt(decoder_hidden_dim), decoder_embeddings.dtype
        )

        # Handle EOI embedding replacement in decoder.
        if not text_only_model:
            dec_eoi_mask = ops.cast(
                ops.expand_dims(
                    ops.equal(decoder_token_id_input, eoi_token_index),
                    axis=-1,
                ),
                decoder_embeddings.dtype,
            )
            decoder_embeddings = (
                dec_eoi_mask * self.decoder_eoi_embedding
                + (1 - dec_eoi_mask) * decoder_embeddings
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

        inputs = {
            "encoder_token_ids": encoder_token_id_input,
            "encoder_padding_mask": encoder_padding_mask_input,
            "decoder_token_ids": decoder_token_id_input,
            "decoder_padding_mask": decoder_padding_mask_input,
        }
        if not text_only_model:
            inputs.update(
                {
                    "images": image_input,
                    "vision_indices": vision_indices_input,
                }
            )

        super().__init__(
            inputs=inputs,
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
        self.global_rope_scaling_factor = global_rope_scaling_factor
        self.encoder_rope_max_wavelength = encoder_rope_max_wavelength
        self.encoder_global_rope_scaling_factor = (
            encoder_global_rope_scaling_factor
        )
        self.use_query_key_norm = use_query_key_norm
        self.eoi_token_index = eoi_token_index
        self.text_only_model = text_only_model

        # Keep `num_vision_tokens_per_image` for easy access.
        if not text_only_model:
            self.num_vision_tokens_per_image = (
                self.vision_encoder.num_vision_tokens_per_image
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_intermediate_dim": self.encoder_intermediate_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_num_attention_heads": (
                    self.encoder_num_attention_heads
                ),
                "encoder_num_key_value_heads": (
                    self.encoder_num_key_value_heads
                ),
                "encoder_layer_types": self.encoder_layer_types,
                "encoder_head_dim": self.encoder_head_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "decoder_intermediate_dim": self.decoder_intermediate_dim,
                "decoder_num_layers": self.decoder_num_layers,
                "decoder_num_attention_heads": (
                    self.decoder_num_attention_heads
                ),
                "decoder_num_key_value_heads": (
                    self.decoder_num_key_value_heads
                ),
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
                "cross_attention_hidden_size": (
                    self.cross_attention_hidden_size
                ),
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "final_logit_softcapping": (self.final_logit_softcapping),
                "rope_max_wavelength": self.rope_max_wavelength,
                "global_rope_scaling_factor": (self.global_rope_scaling_factor),
                "encoder_rope_max_wavelength": (
                    self.encoder_rope_max_wavelength
                ),
                "encoder_global_rope_scaling_factor": (
                    self.encoder_global_rope_scaling_factor
                ),
                "use_query_key_norm": self.use_query_key_norm,
                "eoi_token_index": self.eoi_token_index,
            }
        )
        if self.vision_encoder is not None:
            config["vision_encoder"] = keras.saving.serialize_keras_object(
                self.vision_encoder
            )
        return config

    @classmethod
    def from_config(cls, config):
        vision_encoder = config.pop("vision_encoder", None)
        if vision_encoder is not None and isinstance(vision_encoder, dict):
            vision_encoder = keras.saving.deserialize_keras_object(
                vision_encoder
            )
            config["vision_encoder"] = vision_encoder
        elif vision_encoder is not None:
            config["vision_encoder"] = vision_encoder
        return cls(**config)
