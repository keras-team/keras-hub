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
from absl import logging

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import ops
from keras_nlp.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.src.utils.keras_utils import pack_x_y_sample_weight


@keras_nlp_export("keras_nlp.models.PaliGemmaCausalLMPreprocessor")
class PaliGemmaCausalLMPreprocessor(GemmaCausalLMPreprocessor):
    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer, sequence_length, add_start_token, add_end_token, **kwargs
        )

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`PaliGemmaCausalLMPreprocessor` generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored."
            )
        sequence_length = sequence_length or self.sequence_length

        images, text = x["images"], x["text"]
        images = ops.convert_to_tensor(images)

        x = convert_inputs_to_list_of_tensor_segments(text)[0]
        x = self.tokenizer(x)
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask = self.packer(
            x,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "images": images,
        }
        # Target `y` will be the next token.
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)

    def generate_preprocess(
        self,
        input,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)
        images = input["images"]
        x = input["text"]
        x = convert_inputs_to_list_of_tensor_segments(x)[0]
        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        return {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
