import keras
from keras import layers
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_encoder_block import (  # noqa: E501
    EmbeddingGemma2EncoderBlock,
)
from keras_hub.src.models.gemma4.gemma4_layers import (
    Gemma4InterleaveEmbeddings,  # noqa: E501
)
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization


@keras_hub_export("keras_hub.models.EmbeddingGemma2Backbone")
class EmbeddingGemma2Backbone(Backbone):
    """EmbeddingGemma 2 core network with hyperparameters.

    This backbone implements the EmbeddingGemma 2 encoder: the Gemma 4 text
    stack run with fully bidirectional attention, followed by a dense
    `embedding_projection` applied to every token's final hidden state.
    Mean pooling and L2 normalisation live in
    `keras_hub.models.EmbeddingGemma2TextEmbedder`.

    Images, video frames and audio clips are encoded by the optional
    `vision_encoder` (`Gemma4VisionEncoder`) and `audio_encoder`
    (`Gemma4AudioEncoder`); their soft tokens are interleaved into the text
    embedding sequence at the placeholder positions before the encoder
    blocks run.

    The default constructor gives a fully customised, randomly initialised
    model. To load preset weights use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The spatial resolution of images (height = width).
            Stored as a config value for serialization purposes only; it does
            not affect the backbone's forward pass. Image patching and resizing
            are handled by `keras_hub.layers.EmbeddingGemma2ImageConverter`
            before data reaches the backbone. The `vision_encoder` has its own
            `image_size` parameter that controls position embedding sizes.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query heads per attention layer.
        num_key_value_heads: int. Number of key/value heads (GQA).
        hidden_dim: int. Hidden state dimension at the end of each layer.
        intermediate_dim: int. First dense layer output dimension in each FFW
            sub-block.
        head_dim: int. Per-head dimension in the encoder attention.
        use_sliding_window_attention: bool. Whether to use sliding-window
            attention on the local layers. Defaults to `True`.
        sliding_window_size: int. Size of the local attention window. Defaults
            to `512`.
        sliding_window_pattern: int. Repeat period of the local/global
            attention pattern. The last layer in each group of this many
            consecutive layers uses global attention; all others use local
            (sliding-window) attention. Defaults to `6`.
        layer_types: list of str or `None`. Explicit specification of the
            attention type for every layer sequentially
            (e.g. `"full_attention"`, `"sliding_attention"`). When `None`,
            type sequence is derived from `sliding_window_pattern`.
            Defaults to `None`.
        global_head_dim: int or `None`. Per-head dimension used specifically
            for global attention layers. When `None`, `head_dim` is used
            for all layers. Defaults to `None`.
        local_rope_scaling_factor: float. RoPE scaling factor for local layers.
            Defaults to `1.0`.
        global_rope_scaling_factor: float. RoPE scaling factor for global
            layers. Defaults to `1.0`.
        global_rope_partial_rotary_factor: float. Fraction of each head
            dimension that receives rotary position embeddings in global
            attention layers. Only the first
            `int(factor * head_dim)` dimensions are rotated; the remainder are
            left unchanged (NoPE). Local layers always use full RoPE
            (`factor = 1.0`). Defaults to `1.0`.
        vision_encoder: `keras_hub.models.Gemma4VisionEncoder` or `None`. When
            `None` the model processes no images.
        audio_encoder: `keras_hub.models.Gemma4AudioEncoder` or `None`. When
            `None` the model processes no audio.
        num_audio_tokens_per_clip: int or `None`. Number of audio soft tokens
            produced per audio clip (including zero-padded positions). When
            `None` and `audio_encoder` is set, defaults to `750`.
        layer_norm_epsilon: float. Epsilon for all RMS norms. Defaults `1e-6`.
        use_bidirectional_attention: bool. When `True` the model uses fully
            bidirectional attention for ALL tokens, as required for an
            embedding model. Defaults to `False`.
        dropout: float. Dropout probability. Defaults to `0`.
        embedding_dim: int. Output dimension of the final dense
            `embedding_projection` applied to every token's hidden state.
        num_global_key_value_heads: int or `None`. When set, global attention
            layers use this many K/V heads instead of `num_key_value_heads`
            (separate K and V projections are kept). Defaults to `None`.
        hidden_size_per_layer_input: int. Size of the per-token, per-layer
            conditioning vector that gates each decoder layer's output.
            Set to `0` to disable. Defaults to `0`.

        dtype: string or `keras.mixed_precision.DTypePolicy`. Compute dtype.
            Defaults to `None`.

    Examples:

    ```python
    # Load a pretrained EmbeddingGemma 2 backbone.
    model = keras_hub.models.EmbeddingGemma2Backbone.from_preset(
        "embedding_gemma2"
    )

    # Randomly initialised text-only backbone with a custom config.
    model = keras_hub.models.EmbeddingGemma2Backbone(
        vocabulary_size=262144,
        image_size=896,
        num_layers=24,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=512,
        intermediate_dim=2048,
        head_dim=256,
        embedding_dim=768,
        use_bidirectional_attention=True,
        dtype="float32",
    )
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        use_sliding_window_attention=True,
        sliding_window_size=512,
        sliding_window_pattern=6,
        layer_types=None,
        global_head_dim=None,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        vision_encoder=None,
        audio_encoder=None,
        num_audio_tokens_per_clip=None,
        layer_norm_epsilon=1e-6,
        use_bidirectional_attention=False,
        dropout=0,
        embedding_dim=None,
        num_global_key_value_heads=None,
        hidden_size_per_layer_input=0,
        global_rope_wavelength=None,
        local_rope_wavelength=None,
        global_rope_partial_rotary_factor=1.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
            ),
            dtype=dtype,
            name="token_embedding",
        )

        # Per-layer token-conditioned input.
        # Each decoder layer receives a per-token, per-layer embedding that
        # gates its output.
        if hidden_size_per_layer_input > 0:
            # Projects text embeddings →
            # (num_layers × hidden_size_per_layer_input),
            # scaled by hidden_dim**-0.5.
            self.per_layer_model_projection = keras.layers.Dense(
                num_layers * hidden_size_per_layer_input,
                use_bias=False,
                dtype=dtype,
                name="per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="per_layer_projection_norm",
            )

        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.layer_types = layer_types
        text_only_model = vision_encoder is None and audio_encoder is None
        if vision_encoder is not None:
            self.interleave_embeddings = Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=(
                    self.vision_encoder.num_vision_tokens_per_image
                ),
                dtype=dtype,
                name="interleave_embeddings",
            )
        if audio_encoder is not None:
            if num_audio_tokens_per_clip is None:
                num_audio_tokens_per_clip = 750
            self.audio_interleave_embeddings = Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=num_audio_tokens_per_clip,
                dtype=dtype,
                name="audio_interleave_embeddings",
            )

        # Build transformer layers.
        # Pattern: every 6th layer (index % 6 == 5) is global attention;
        # the rest use (optional) sliding-window local attention.
        # Precompute KV-sharing indices.
        # The last `num_kv_shared_layers` layers reuse K/V from the most
        # recent non-shared layer of the same attention type.
        self.transformer_layers = []
        for i in range(num_layers):
            # A layer is global when it's the last in each group of
            # `sliding_window_pattern` consecutive layers.
            if layer_types is not None:
                is_global = layer_types[i] == "full_attention"
            else:
                is_global = (i % sliding_window_pattern) == (
                    sliding_window_pattern - 1
                )
            sliding_window = use_sliding_window_attention and not is_global
            rope_wavelength = (
                (global_rope_wavelength or 1_000_000.0)
                if is_global
                else (local_rope_wavelength or 10_000.0)
            )
            rope_scaling_factor = (
                global_rope_scaling_factor
                if is_global
                else local_rope_scaling_factor
            )
            # Global attention layers use `num_global_key_value_heads` KV
            # heads but, unlike Gemma 4, keep a separate V projection.
            layer_kv_heads = (
                num_global_key_value_heads
                if is_global and num_global_key_value_heads is not None
                else num_key_value_heads
            )
            # Global layers use proportional (partial) RoPE; local layers get
            # the full RoPE (factor = 1.0).
            layer_rope_partial = (
                global_rope_partial_rotary_factor if is_global else 1.0
            )
            layer_rope_wavelength = rope_wavelength
            layer = EmbeddingGemma2EncoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=layer_kv_heads,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                rope_wavelength=layer_rope_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rope_partial_rotary_factor=layer_rope_partial,
                use_bidirectional_attention=use_bidirectional_attention,
                is_global_attention=is_global,
                global_head_dim=global_head_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                hidden_size_per_layer_input=hidden_size_per_layer_input,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)

        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (i % sliding_window_pattern) == (sliding_window_pattern - 1)
                else "sliding_attention"
                for i in range(num_layers)
            ]

        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # === Functional Model ===

        # Vision inputs.
        # Audio inputs.
        if audio_encoder is not None:
            audio_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_indices"
            )
            audio_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="audio_mask"
            )
            audio_mel_input = keras.Input(
                shape=(None, None, audio_encoder.input_feat_size),
                name="audio_mel",
            )
            audio_mel_mask_input = keras.Input(
                shape=(None, None), dtype="int32", name="audio_mel_mask"
            )

        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Vision inputs.
        if vision_encoder is not None:
            pixel_position_ids_input = keras.Input(
                shape=(None, None, 2), dtype="int32", name="pixel_position_ids"
            )
            pixel_values_input = keras.Input(
                shape=(None, None, None),
                name="pixel_values",
            )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )

        if vision_encoder is not None:
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )

        # Text embeddings.
        text_embeddings = self.token_embedding(token_id_input)

        # Interleave image embeddings. Pre-scale by 1/sqrt(hidden_dim) so that
        # after the global x *= sqrt(hidden_dim) below, vision positions remain
        # at their natural (unscaled) embed_vision magnitude.
        if vision_encoder is not None:
            img_embeddings = self.vision_encoder(
                {
                    "pixel_values": pixel_values_input,
                    "pixel_position_ids": pixel_position_ids_input,
                }
            )
            img_embeddings = img_embeddings * ops.cast(
                float(hidden_dim) ** -0.5, img_embeddings.dtype
            )
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices_input,
            )
        else:
            x = text_embeddings

        # Interleave audio embeddings (same pre-scaling as vision).
        if audio_encoder is not None:
            audio_embeddings = self.audio_encoder(
                audio_mel_input,
                ops.cast(audio_mel_mask_input, "bool"),
            )
            audio_embeddings = audio_embeddings * ops.cast(
                float(hidden_dim) ** -0.5, audio_embeddings.dtype
            )
            x = self.audio_interleave_embeddings(
                image_embeddings=audio_embeddings,
                text_embeddings=x,
                vision_indices=audio_indices_input,
            )

        # Force connection of audio_mask_input if not used in per-layer
        # embeddings
        if audio_encoder is not None and hidden_size_per_layer_input <= 0:
            dummy = ops.cast(audio_mask_input, x.dtype) * 0.0
            dummy = ops.expand_dims(dummy, axis=-1)
            x = x + dummy

        # Per-layer model projection, computed after the global scale.

        # Global scale: text positions → token_embedding * sqrt(hidden_dim);
        # vision/audio positions remain at their pre-scaled embed magnitude.
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)

        # Per-layer model projection, computed after the global scale.
        if hidden_size_per_layer_input > 0:
            _per_proj = self.per_layer_model_projection(x)
            _per_proj = _per_proj * ops.cast(
                float(hidden_dim) ** -0.5, _per_proj.dtype
            )
            per_layer_proj_flat = _per_proj
        else:
            per_layer_proj_flat = None

        # Decoder layers.
        _hpl = hidden_size_per_layer_input
        for i, transformer_layer in enumerate(self.transformer_layers):
            if per_layer_proj_flat is not None:
                proj_i = per_layer_proj_flat[:, :, i * _hpl : (i + 1) * _hpl]
                per_layer_input_i = self.per_layer_projection_norm(proj_i)
            else:
                per_layer_input_i = None

            x, new_cache = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                vision_mask=(
                    None if vision_encoder is None else vision_mask_input
                ),
                per_layer_input=per_layer_input_i,
                positions=None,
            )

        sequence_output = self.layer_norm(x)

        sequence_output = layers.Dense(
            embedding_dim,
            dtype=dtype,
            name="embedding_projection",
            use_bias=False,
        )(sequence_output)

        outputs = sequence_output

        inputs = {
            "padding_mask": padding_mask_input,
            "token_ids": token_id_input,
        }

        if vision_encoder is not None:
            inputs.update(
                {
                    "pixel_position_ids": pixel_position_ids_input,
                    "pixel_values": pixel_values_input,
                    "vision_indices": vision_indices_input,
                    "vision_mask": vision_mask_input,
                }
            )
        if audio_encoder is not None:
            inputs.update(
                {
                    "audio_indices": audio_indices_input,
                    "audio_mask": audio_mask_input,
                    "audio_mel": audio_mel_input,
                    "audio_mel_mask": audio_mel_mask_input,
                }
            )

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.global_head_dim = global_head_dim
        self.local_rope_scaling_factor = local_rope_scaling_factor
        self.global_rope_scaling_factor = global_rope_scaling_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.num_audio_tokens_per_clip = num_audio_tokens_per_clip
        self.num_global_key_value_heads = num_global_key_value_heads
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.global_rope_wavelength = global_rope_wavelength
        self.local_rope_wavelength = local_rope_wavelength
        self.global_rope_partial_rotary_factor = (
            global_rope_partial_rotary_factor
        )

        # Keep `num_vision_tokens_per_image` and `text_only_model` accessible.
        self.num_vision_tokens_per_image = None
        if vision_encoder is not None:
            self.num_vision_tokens_per_image = (
                self.vision_encoder.num_vision_tokens_per_image
            )
        self.text_only_model = text_only_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "sliding_window_pattern": self.sliding_window_pattern,
                "layer_types": self.layer_types,
                "global_head_dim": self.global_head_dim,
                "local_rope_scaling_factor": self.local_rope_scaling_factor,
                "global_rope_scaling_factor": self.global_rope_scaling_factor,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
                "audio_encoder": None
                if self.audio_encoder is None
                else keras.layers.serialize(self.audio_encoder),
                "num_audio_tokens_per_clip": self.num_audio_tokens_per_clip,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "embedding_dim": self.embedding_dim,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "global_rope_wavelength": self.global_rope_wavelength,
                "local_rope_wavelength": self.local_rope_wavelength,
                "global_rope_partial_rotary_factor": (
                    self.global_rope_partial_rotary_factor
                ),
            }
        )
        return config

    def default_lora_layer_names(self):
        target_names = super().default_lora_layer_names()
        if not self.text_only_model:
            target_names += ["query_proj", "value_proj"]
        return target_names

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "vision_encoder": None
                if config.get("vision_encoder") is None
                else keras.layers.deserialize(config["vision_encoder"]),
                "audio_encoder": None
                if config.get("audio_encoder") is None
                else keras.layers.deserialize(config["audio_encoder"]),
            }
        )
        return super().from_config(config)
