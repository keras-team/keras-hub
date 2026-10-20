from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma4.gemma4_decoder_block import (
    Gemma4TextDecoderBlock,
)


def compute_embedding_gemma2_attention_masks(
    x, padding_mask, sliding_window_size
):
    decoder_mask = merge_padding_and_attention_mask(
        inputs=x, padding_mask=padding_mask, attention_mask=None
    )

    batch_size = ops.shape(x)[0]
    input_length = ops.shape(x)[1]

    # Bidirectional attention only
    mask_1 = decoder_mask
    if mask_1 is None:
        # Attend to everything by default if no padding
        full_mask = ops.ones(
            (batch_size, input_length, input_length), dtype="bool"
        )
    else:
        full_mask = ops.cast(mask_1, "bool")
        query_mask = ops.expand_dims(ops.cast(padding_mask, "bool"), axis=-1)
        full_mask = ops.logical_and(full_mask, query_mask)

    # Sliding-window layers use a symmetric, *inclusive* window, matching
    # HF `masking_utils.sliding_window_bidirectional_overlay`
    # (`abs(q_idx - kv_idx) <= sliding_window`).
    i = ops.expand_dims(ops.arange(input_length, dtype="int32"), axis=1)
    j = ops.expand_dims(ops.arange(input_length, dtype="int32"), axis=0)
    window_mask = ops.abs(i - j) <= sliding_window_size
    sliding_mask = full_mask & window_mask

    return full_mask, sliding_mask


class EmbeddingGemma2EncoderBlock(Gemma4TextDecoderBlock):
    """An encoder block for EmbeddingGemma2 models.

    This block subclasses `Gemma4TextDecoderBlock` to override the attention
    mask computation. EmbeddingGemma2 uses bidirectional attention. For sliding
    layers, it restricts attention to a symmetric window around the target
    token: `abs(i - j) <= sliding_window_size`. For full layers, it attends to
    all tokens symmetrically.
    """

    def __init__(self, **kwargs):
        # EmbeddingGemma2 uses bidirectional attention.
        kwargs["use_bidirectional_attention"] = True
        super().__init__(**kwargs)

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        vision_mask,
        cache,
        cache_update_index,
    ):
        full_mask, sliding_mask = compute_embedding_gemma2_attention_masks(
            x, padding_mask, self.sliding_window_size
        )
        if self.use_sliding_window_attention and not self.is_global_attention:
            return sliding_mask
        return full_mask
