# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_nlp.src.backend import ops
from keras_nlp.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_nlp.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_nlp.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock


class PaliGemmaDecoderBlock(GemmaDecoderBlock):
    def __init__(
        self,
        img_sequence_length,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        layer_norm_epsilon=1e-6,
        dropout=0,
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            **kwargs,
        )
        self.img_sequence_length = img_sequence_length

    def _compute_attention_mask(
        self, x, padding_mask, cache, cache_update_index
    ):
        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        complete_padding_mask = None
        if padding_mask is not None:
            complete_padding_mask = ops.concatenate(
                [
                    ops.full((batch_size, self.img_sequence_length), True),
                    padding_mask,
                ],
                axis=1,
            )

        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=complete_padding_mask, attention_mask=None
        )
        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )
        # Image sequence embeddings should be fully self-attended without
        # a causal mask.
        text_sequence_length = input_length - self.img_sequence_length
        img_causal_mask = ops.concatenate(
            [
                ops.ones((batch_size, output_length, self.img_sequence_length)),
                ops.zeros((batch_size, output_length, text_sequence_length)),
            ],
            axis=-1,
        )
        causal_mask = ops.maximum(causal_mask, img_causal_mask)

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_sequence_length": self.img_sequence_length,
            }
        )
        return config
