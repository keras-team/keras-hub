import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock


class PaliGemmaDecoderBlock(GemmaDecoderBlock):
    """PaliGemma mixed decoder block.

    This class implements a decoder block of the PaliGemma Architecture: a
    mixed transformer decoder block. Intended to be used with an input
    sequence comprised of both embedded image and text data, this block
    functions largely identically to the `GemmaDecoderBlock` class, with a
    notable exception in the computation of attention masks.

    Specifically, this decoder block will use causal self-attention on the
    text portion of the input, while using full self-attention for image
    data. It is expected that any image data occurs before text data in the
    input.

    Args:
        hidden_dim: int. The size of the transformer hidden state at the end
            of the block.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the two-layer feedforward network.
        head_dim: int. The size of each attention head.
        num_query_heads: int. The number of heads for the query projections in
            the attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the attention layer.
        query_head_dim_normalize: boolean. If `True` normalize the query before
            attention with `head_dim`. If `False`, normalize the query with
            `hidden_dim / num_query_heads`. Defaults to `True`.
        use_post_ffw_norm: boolean. Whether to normalize after the feedforward
            block. Defaults to `False`.
        use_post_attention_norm: boolean. Whether to normalize after the
            attention block. Defaults to `False`.
        logit_soft_cap: `None` or int. Soft cap for the attention logits.
            Defaults to `None`.
        use_sliding_window_attention: boolean. Whether to use sliding local
          window attention. Defaults to `False`.
        sliding_window_size: int. Size of the sliding local window. Defaults to
            `4096`.
        layer_norm_epsilon: float. The epsilon hyperparameter used for layer
            normalization. Defaults to `1e-6`.
        dropout: float. The dropout rate for the transformer attention layer.
            Defaults to `0`.
    """

    def call(
        self,
        x,
        padding_mask=None,
        response_mask=None,
        cache=None,
        cache_update_index=0,
    ):
        normalized_x = self.pre_attention_norm(x)
        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, cache, cache_update_index, response_mask
        )
        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
            )
        else:
            attention = self.attention(
                normalized_x,
                attention_mask=attention_mask,
            )

        if self.use_post_attention_norm:
            attention = self.post_attention_norm(attention)

        if self.dropout:
            attention = self.attention_dropout(attention)

        attention_x = x + attention
        normalized_x = self.pre_ffw_norm(attention_x)

        x1 = self.gating_ffw(normalized_x)
        x2 = self.gating_ffw_2(normalized_x)
        x = keras.activations.gelu(x1, approximate=True) * x2
        x = self.ffw_linear(x)

        if self.use_post_ffw_norm:
            x = self.post_ffw_norm(x)

        x = x + attention_x

        if cache is not None:
            return x, new_cache
        return x

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        cache,
        cache_update_index,
        response_mask=None,
    ):
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

        if padding_mask is None:
            # We should only hit this case during generative decoding.
            # Just the causal mask is fine in this case.
            return causal_mask

        def token_to_attention_mask(mask, fill_value):
            """Reshape token mask -> attention mask padding for image tokens."""
            mask = ops.cast(mask, "int32")
            pad = input_length - ops.shape(mask)[1]
            mask = ops.pad(mask, ((0, 0), (pad, 0)), constant_values=fill_value)
            return ops.expand_dims(mask, axis=1)

        padding_mask = token_to_attention_mask(padding_mask, 1)
        if response_mask is not None:
            response_mask = token_to_attention_mask(response_mask, 0)
            not_response_mask = ops.logical_not(response_mask)
            # Only apply the causal mask to the response tokens.
            causal_mask = ops.logical_and(causal_mask, response_mask)
            # Only apply block attention to the non-response tokens.
            padding_mask = ops.logical_and(padding_mask, not_response_mask)

        # Use block attention for the padding mask,
        # which marks all image and prompt tokens.
        return ops.logical_or(padding_mask, causal_mask)
