# Copyright 2024 The KerasHub Authors
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
import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.models.pali_gemma.pali_gemma_image_converter import (
    PaliGemmaImageConverter,
)
from keras_hub.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.PaliGemmaCausalLMPreprocessor")
class PaliGemmaCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = PaliGemmaBackbone
    tokenizer_cls = PaliGemmaTokenizer
    image_converter_cls = PaliGemmaImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images, prompts, responses = x["images"], x["prompts"], x["responses"]
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)
        if self.image_converter:
            images = self.image_converter(images)
        # Pad with one extra token to account for the truncation below.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "response_mask": response_mask[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "images": images,
        }
        # Target `y` will be the next token.
        y = token_ids[..., 1:]
        # Only compute the loss for labels in the response.
        sample_weight = response_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
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
        sequence_length = sequence_length or self.sequence_length

        images, prompts = x["images"], x["prompts"]
        prompts = self.tokenizer(prompts)
        if self.image_converter:
            images = self.image_converter(images)
        if "responses" in x:
            responses = self.tokenizer(x["responses"])
            segments = (prompts, responses)
        else:
            segments = (prompts,)
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1
        return {
            "images": images,
            "token_ids": token_ids,
            "response_mask": response_mask,
            "padding_mask": padding_mask,
        }
