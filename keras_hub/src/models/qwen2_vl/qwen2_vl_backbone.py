import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_decoder import QwenTransformerDecoder
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionEncoder,
)


def _qwen2vl_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen2VLBackbone")
class Qwen2VLBackbone(Backbone):
    """Qwen2-VL multimodal backbone.

    Combines a 3D Vision Encoder (ViT with RoPE + PatchMerger) with a
    Qwen2 causal language model decoder. Vision tokens produced by the
    encoder replace the ``image_token_id`` placeholder tokens in the text
    sequence before being passed through the decoder layers.

    Args:
        vocabulary_size: int. Vocabulary size of the text model.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads (GQA).
        hidden_dim: int. LLM hidden dimension.
        intermediate_dim: int. Feed-forward intermediate dimension.
        vision_patch_size: int. Spatial patch size for the vision encoder.
            Defaults to ``14``.
        vision_temporal_patch_size: int. Temporal patch size. Defaults to
            ``2``.
        vision_in_channels: int. Vision input channels. Defaults to ``3``.
        vision_embed_dim: int. Vision encoder internal dimension. Defaults
            to ``1280``.
        vision_depth: int. Number of vision transformer blocks. Defaults to
            ``32``.
        vision_num_heads: int. Vision attention heads. Defaults to ``16``.
        vision_mlp_ratio: float. Vision MLP hidden dim multiplier. Defaults
            to ``4``.
        spatial_merge_size: int. Spatial merge factor for PatchMerger.
            Defaults to ``2``.
        image_token_id: int. Token id used as image placeholder in the text
            sequence. The number of ``image_token_id`` placeholders in the
            input must exactly equal the number of merged vision tokens
            produced by encoding ``patch_values`` with ``image_grid_thw``.
            Defaults to ``151655``.
        rope_max_wavelength: int. RoPE base wavelength for the text model.
            Defaults to ``1000000``.
        rope_scaling_factor: float. RoPE scaling factor. Defaults to ``1.0``.
        layer_norm_epsilon: float. Epsilon for RMS norm layers. Defaults to
            ``1e-6``.
        dropout: float. Dropout rate. Defaults to ``0``.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
            Defaults to ``False``.
        use_sliding_window_attention: bool. Whether to use sliding window
            attention. Defaults to ``False``.
        sliding_window_size: int. Sliding window size. Defaults to ``32768``.
        dtype: string or ``keras.mixed_precision.DTypePolicy``.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        vision_patch_size=14,
        vision_temporal_patch_size=2,
        vision_in_channels=3,
        vision_embed_dim=1280,
        vision_depth=32,
        vision_num_heads=16,
        vision_mlp_ratio=4,
        spatial_merge_size=2,
        image_token_id=151655,
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0,
        tie_word_embeddings=False,
        use_sliding_window_attention=False,
        sliding_window_size=32768,
        dtype=None,
        **kwargs,
    ):
        # === Vision encoder ===
        self.vision_encoder = Qwen2VLVisionEncoder(
            patch_size=vision_patch_size,
            temporal_patch_size=vision_temporal_patch_size,
            in_channels=vision_in_channels,
            embed_dim=vision_embed_dim,
            hidden_size=hidden_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            spatial_merge_size=spatial_merge_size,
            dtype=dtype,
            name="vision_encoder",
        )

        # === Text decoder ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen2vl_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = QwenTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen2vl_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                use_sliding_window_attention=use_sliding_window_attention,
                sliding_window_size=sliding_window_size,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # === Functional model ===
        # Only text inputs in functional graph; vision inputs handled in call()
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Text embedding path (vision merging happens in call())
        token_embeddings = self.token_embedding(token_ids)
        x = token_embeddings
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask)
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_patch_size = vision_patch_size
        self.vision_temporal_patch_size = vision_temporal_patch_size
        self.vision_in_channels = vision_in_channels
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size

    def call(self, inputs, training=None):
        """Forward pass with vision token replacement.

        Embeds text tokens, encodes vision patches, replaces
        ``image_token_id`` placeholder positions in the embedding sequence
        with the merged vision features, then runs the decoder.

        Args:
            inputs: Dict with keys ``"token_ids"``, ``"padding_mask"``,
                ``"patch_values"`` (optional), ``"image_grid_thw"``
                (optional).
            training: bool or None.

        Returns:
            Hidden-state tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        patch_values = inputs.get("patch_values", None)
        grid_thw = inputs.get("image_grid_thw", None)

        # Embed text tokens → (batch, seq_len, hidden_dim)
        x = self.token_embedding(token_ids)

        # If vision inputs are present, encode and scatter into x.
        if patch_values is not None and grid_thw is not None:
            # vision_features: (total_merged_tokens, hidden_dim)
            vision_features = self.vision_encoder(
                patch_values, grid_thw, training=training
            )
            # Build a boolean mask of image placeholder positions.
            # image_mask: (batch, seq_len)
            image_mask = ops.equal(
                token_ids,
                ops.cast(self.image_token_id, token_ids.dtype),
            )
            # Flatten batch+seq dims, replace masked positions with
            # vision features, then restore shape.
            batch_size = ops.shape(x)[0]
            seq_len = ops.shape(x)[1]
            x_flat = ops.reshape(x, (-1, self.hidden_dim))
            mask_flat = ops.reshape(image_mask, (-1,))
            # vision_features is already in the right order (same order as
            # the image placeholder tokens appear left-to-right).
            vision_indices = ops.where(mask_flat)
            if isinstance(vision_indices, (list, tuple)):
                vision_indices = vision_indices[0]
            vision_indices = ops.reshape(vision_indices, (-1, 1))
            vision_indices = ops.cast(vision_indices, "int32")
            n_placeholders = ops.shape(vision_indices)[0]
            n_vision = ops.shape(vision_features)[0]
            if n_placeholders != n_vision:
                raise ValueError(
                    f"Vision token count mismatch: the number of "
                    f"image_token_id={self.image_token_id} placeholders "
                    f"in token_ids ({n_placeholders}) does not equal the "
                    f"number of merged vision tokens produced by the "
                    f"vision encoder from patch_values/image_grid_thw "
                    f"({n_vision}). Ensure the preprocessor inserts "
                    f"exactly one placeholder per merged vision token."
                )
            x_flat = ops.scatter_update(x_flat, vision_indices, vision_features)
            x = ops.reshape(x_flat, (batch_size, seq_len, self.hidden_dim))

        # Decoder layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                decoder_padding_mask=padding_mask,
                training=training,
            )

        # Final layer norm
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_temporal_patch_size": self.vision_temporal_patch_size,
                "vision_in_channels": self.vision_in_channels,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_depth": self.vision_depth,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "spatial_merge_size": self.spatial_merge_size,
                "image_token_id": self.image_token_id,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
