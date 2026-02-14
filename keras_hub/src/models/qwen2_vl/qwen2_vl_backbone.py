"""Qwen2-VL Backbone model.

Integrates the Vision Transformer encoder with the text decoder,
supporting multimodal (image + text) and text-only inputs.
"""

import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_decoder import (
    Qwen2VLTransformerDecoder,
)
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm


def _qwen2_vl_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Qwen2VLBackbone")
class Qwen2VLBackbone(Backbone):
    """Qwen2-VL core network with optional vision encoder.

    This implements the Qwen2-VL model architecture combining a Vision
    Transformer with a Transformer text decoder via Multimodal RoPE
    (M-RoPE). The backbone supports both vision+text and text-only modes.

    When a `vision_encoder` is provided, images are encoded and their
    embeddings are interleaved into the text embedding sequence at
    positions specified by `vision_indices`. Multimodal position IDs
    (`mrope_position_ids`) are used to compute M-RoPE embeddings.

    The default constructor gives a fully customizable, randomly
    initialized model. Use `from_preset()` to load preset architectures.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads (GQA).
        hidden_dim: int. Hidden dimension of the transformer.
        intermediate_dim: int. Intermediate dimension of the MLP.
        mrope_section: list. List of 3 ints specifying the M-RoPE section
            sizes `[temporal, height, width]` in half-head-dim units.
        rope_max_wavelength: float. Max wavelength for RoPE.
            Defaults to `10000`.
        layer_norm_epsilon: float. Epsilon for RMS norm.
            Defaults to `1e-6`.
        dropout: float. Dropout rate. Defaults to `0`.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
            Defaults to `True`.
        vision_encoder: optional. A `Qwen2VLVisionEncoder` instance.
            If `None`, the model operates in text-only mode.
        dtype: string or `keras.mixed_precision.DTypePolicy`.

    Call arguments:
        token_ids: Tensor of token IDs, shape `(batch, seq_len)`.
        padding_mask: Tensor, shape `(batch, seq_len)`.
        images: Optional tensor of pixel values for the vision encoder.
        vision_indices: Optional tensor of positions where image
            embeddings should be placed, shape
            `(batch, num_vision_tokens)`.
        vision_mask: Optional boolean mask, shape `(batch, seq_len)`,
            indicating vision token positions.
        mrope_position_ids: Tensor of shape `(3, batch, seq_len)` with
            M-RoPE position IDs (temporal, height, width).

    Example:
    ```python
    from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
        Qwen2VLVisionEncoder,
    )
    from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import (
        Qwen2VLBackbone,
    )

    # Text-only backbone
    backbone = Qwen2VLBackbone(
        vocabulary_size=151936,
        num_layers=4,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=64,
        intermediate_dim=128,
        mrope_section=[2, 2, 2],
    )

    # Vision + text backbone
    vision_encoder = Qwen2VLVisionEncoder(
        hidden_size=64,
        embed_dim=32,
        depth=2,
        num_heads=2,
        patch_size=14,
    )
    backbone = Qwen2VLBackbone(
        vocabulary_size=151936,
        num_layers=4,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=64,
        intermediate_dim=128,
        mrope_section=[2, 2, 2],
        vision_encoder=vision_encoder,
    )
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        mrope_section,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-6,
        dropout=0,
        tie_word_embeddings=True,
        vision_encoder=None,
        dtype=None,
        **kwargs,
    ):
        # Determine text-only mode
        text_only_model = vision_encoder is None
        head_dim = hidden_dim // num_query_heads

        # === Layers ===
        token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            embeddings_initializer=_qwen2_vl_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )

        transformer_layers = []
        for i in range(num_layers):
            layer = Qwen2VLTransformerDecoder(
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                mrope_section=mrope_section,
                rope_max_wavelength=rope_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_qwen2_vl_kernel_initializer(stddev=0.02),
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            transformer_layers.append(layer)

        layer_norm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        mrope_position_ids_input = keras.Input(
            shape=(3, None), dtype="int32", name="mrope_position_ids"
        )

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "mrope_position_ids": mrope_position_ids_input,
        }

        # Token embedding
        text_embeddings = token_embedding(token_id_input)

        if not text_only_model:
            # Vision inputs
            image_input = keras.Input(
                shape=(None,),
                dtype="float32",
                name="images",
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )
            grid_thw_input = keras.Input(
                shape=(None, 3), dtype="int32", name="grid_thw"
            )

            inputs["images"] = image_input
            inputs["vision_indices"] = vision_indices_input
            inputs["vision_mask"] = vision_mask_input
            inputs["grid_thw"] = grid_thw_input

        # Compute M-RoPE position embeddings
        # mrope_position_ids shape: (batch, 3, seq_len)
        # We need cos/sin of shape (3, batch, seq_len, head_dim)
        position_embeddings = _compute_mrope_embeddings(
            mrope_position_ids_input,
            head_dim,
            rope_max_wavelength,
            mrope_section,
        )

        x = text_embeddings

        # Decoder layers
        for transformer_layer in transformer_layers:
            x = transformer_layer(
                x,
                attention_mask=padding_mask_input,
                position_embeddings=position_embeddings,
            )

        sequence_output = layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Store config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_encoder = vision_encoder
        self.token_embedding = token_embedding
        self.transformer_layers = transformer_layers
        self.layer_norm = layer_norm

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
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "vision_encoder": (
                    keras.layers.serialize(self.vision_encoder)
                    if self.vision_encoder is not None
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        vision_encoder = config.pop("vision_encoder", None)
        if vision_encoder is not None:
            vision_encoder = keras.layers.deserialize(vision_encoder)
        return cls(vision_encoder=vision_encoder, **config)


def _compute_mrope_embeddings(
    mrope_position_ids, head_dim, rope_max_wavelength, mrope_section
):
    """Compute M-RoPE cos/sin embeddings from position IDs.

    Args:
        mrope_position_ids: Tensor of shape `(batch, 3, seq_len)` with
            [temporal, height, width] position indices.
        head_dim: int. Head dimension.
        rope_max_wavelength: float. Base wavelength for RoPE.
        mrope_section: list. Section sizes for [t, h, w].

    Returns:
        Tuple of (cos, sin), each of shape
        `(3, batch, seq_len, head_dim)`.
    """
    # Compute inverse frequencies
    dim = head_dim
    inv_freq = 1.0 / (
        rope_max_wavelength
        ** (ops.cast(ops.arange(0, dim, 2), "float32") / dim)
    )

    # mrope_position_ids: (batch, 3, seq_len)
    # Transpose to (3, batch, seq_len) for each component
    position_ids = ops.transpose(
        ops.cast(mrope_position_ids, "float32"), (1, 0, 2)
    )

    # inv_freq_expanded: (1, 1, dim//2, 1)
    inv_freq_expanded = ops.reshape(inv_freq, (1, 1, -1, 1))
    # Expand to (3, batch_size, dim//2, 1)
    inv_freq_expanded = ops.broadcast_to(
        inv_freq_expanded,
        (3, ops.shape(position_ids)[1], len(inv_freq), 1),
    )

    # position_ids_expanded: (3, batch, 1, seq_len)
    position_ids_expanded = ops.expand_dims(position_ids, axis=2)

    # freqs: (3, batch, dim//2, seq_len) -> transpose -> (3, batch, seq_len, dim//2)
    freqs = ops.matmul(
        ops.cast(inv_freq_expanded, "float32"),
        ops.cast(position_ids_expanded, "float32"),
    )
    freqs = ops.transpose(freqs, (0, 1, 3, 2))

    # Concatenate to get full head_dim: (3, batch, seq_len, head_dim)
    emb = ops.concatenate([freqs, freqs], axis=-1)

    cos = ops.cos(emb)
    sin = ops.sin(emb)

    return (cos, sin)
