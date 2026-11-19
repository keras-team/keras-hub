from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


@keras_hub_export("keras_hub.models.Qwen2_5VLBackbone")
class Qwen2_5VLBackbone(Qwen2VLBackbone):
    """Qwen2.5-VL multimodal backbone.

    A 3D vision encoder produces patch features that are merged into the
    text embedding stream at ``vision_indices``; the Qwen2.5 decoder then
    applies multimodal RoPE (M-RoPE) from three-channel ``position_ids``
    (temporal, height, width). The text-side architecture is unchanged
    from ``Qwen2VLBackbone`` — the Qwen2.5-VL deltas live in
    ``Qwen2_5VLVisionEncoder`` (windowed attention selected via
    ``fullatt_block_indexes``, RMSNorm, gated SiLU MLP) and in the
    preprocessor's absolute-time temporal positions driven by per-video
    ``second_per_grid_ts``.

    Args:
        vocabulary_size: int. Vocabulary size of the text model.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads (GQA).
        hidden_dim: int. LLM hidden dimension.
        intermediate_dim: int. Feed-forward intermediate dimension.
        vision_encoder: A ``Qwen2_5VLVisionEncoder`` instance, or ``None``
            for a text-only model.
        mrope_section: list of int or None. M-RoPE section sizes
            ``[s_t, s_h, s_w]`` in rotary pairs, summing to
            ``head_dim // 2``. Defaults to ``[16, 24, 24]`` (the value
            used by released Qwen2.5-VL checkpoints, whose ``head_dim``
            is 128).
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
