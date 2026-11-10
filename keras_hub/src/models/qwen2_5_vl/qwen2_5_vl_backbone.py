from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


@keras_hub_export("keras_hub.models.Qwen2_5VLBackbone")
class Qwen2_5VLBackbone(Qwen2VLBackbone):
    """Qwen2.5-VL multimodal backbone.

    Identical to ``Qwen2VLBackbone``: the Qwen2.5-VL text decoder and
    M-RoPE wiring are unchanged. The architecture delta lives entirely in
    ``Qwen2_5VLVisionEncoder`` (windowed attention, RMSNorm, gated SiLU
    MLP) and in the preprocessor's time-aware video positions; pass a
    ``Qwen2_5VLVisionEncoder`` instance as ``vision_encoder``.

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
            ``head_dim // 2``. Defaults to ``[16, 24, 24]``.
        rope_max_wavelength: int. RoPE base wavelength for the text model.
        rope_scaling_factor: float. RoPE scaling factor.
        layer_norm_epsilon: float. Epsilon for RMS norm layers.
        dropout: float. Dropout rate.
        tie_word_embeddings: bool. Whether to tie input/output embeddings.
        use_sliding_window_attention: bool. Whether to use sliding window
            attention.
        sliding_window_size: int. Sliding window size.
        dtype: string or ``keras.mixed_precision.DTypePolicy``.
    """
