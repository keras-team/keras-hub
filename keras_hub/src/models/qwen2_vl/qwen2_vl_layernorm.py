from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm


class Qwen2VLLayerNorm(QwenLayerNorm):
    """Qwen2-VL RMS LayerNorm.

    Reuses the existing ``QwenLayerNorm`` implementation (RMS
    normalization without centering).

    Args:
        epsilon: float. A small float added to the denominator to
            avoid dividing by zero. Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The
            dtype for computations and weights.
    """

    pass
