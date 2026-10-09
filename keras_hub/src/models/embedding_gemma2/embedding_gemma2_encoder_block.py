from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma4.gemma4_decoder_block import (
    Gemma4TextDecoderBlock,
)


@keras_hub_export("keras_hub.models.EmbeddingGemma2EncoderBlock")
class EmbeddingGemma2EncoderBlock(Gemma4TextDecoderBlock):
    """An encoder block for EmbeddingGemma2 models.

    This block subclasses `Gemma4TextDecoderBlock` to override the attention
    mask computation. EmbeddingGemma2 uses bidirectional attention. For sliding
    layers, it restricts attention to a symmetric window around the target
    token: `abs(i - j) <= sliding_window_size`. For full layers, it attends to
    all tokens symmetrically.
    """

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        vision_mask,
        cache,
        cache_update_index,
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
            mask = ops.ones(
                (batch_size, input_length, input_length), dtype="bool"
            )
        else:
            mask_2 = ops.transpose(mask_1, (0, 2, 1))
            mask = ops.cast(mask_1 * mask_2, "bool")

        # Sliding-window layers use a symmetric, *inclusive* window, matching
        # HF `masking_utils.sliding_window_bidirectional_overlay`
        # (`abs(q_idx - kv_idx) <= sliding_window`).
        if self.use_sliding_window_attention and not self.is_global_attention:
            i = ops.expand_dims(ops.arange(input_length), axis=1)
            j = ops.expand_dims(ops.arange(input_length), axis=0)
            window_mask = ops.abs(i - j) <= self.sliding_window_size
            mask = mask & window_mask

        return mask
