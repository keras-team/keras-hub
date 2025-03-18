from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock


class Gemma3DecoderBlock(GemmaDecoderBlock):
    def _compute_image_bidirectional_attention_mask(self, text_mask):
        # text_mask is True for text, False for images. Shape of (bsz, seq_len).
        bidirectional_mask = ops.logical_not(text_mask)

        # Left pad with 0.
        padded_mask = ops.cast(
            ops.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0),
            dtype="int32",
        )

        # Assign unique indices to every contiguous span of True.
        boundary = ops.cast(
            ops.greater(padded_mask[..., 1:], padded_mask[..., :-1]),
            dtype="int32",
        )
        numbered_boundary = ops.cumsum(boundary, -1)
        indices = ops.multiply(bidirectional_mask, numbered_boundary)

        indices_expanded_1 = ops.expand_dims(indices, 1)
        indices_expanded_2 = ops.expand_dims(indices, -1)

        mask = ops.logical_and(
            ops.equal(
                indices_expanded_1,
                indices_expanded_2,
            ),
            indices_expanded_2,
        )
        return mask

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        text_mask,
        cache,
        cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )

        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )
        if decoder_mask is not None:
            causal_mask = ops.minimum(decoder_mask, causal_mask)

        # Compute bidirectional mask (image tokens can attend to each other
        # in both directions, within the same image).
        if text_mask is not None:
            bidirectional_image_mask = (
                self._compute_image_bidirectional_attention_mask(text_mask)
            )
            causal_mask = ops.logical_or(causal_mask, bidirectional_image_mask)
        return causal_mask
