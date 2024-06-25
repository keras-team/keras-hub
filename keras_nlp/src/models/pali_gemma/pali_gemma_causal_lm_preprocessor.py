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
import keras
from absl import logging

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_nlp.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)


@keras_nlp_export("keras_nlp.models.PaliGemmaCausalLMPreprocessor")
class PaliGemmaCausalLMPreprocessor(GemmaCausalLMPreprocessor):
    tokenizer_cls = PaliGemmaTokenizer

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

        images, prompts, responses = x["images"], x["prompts"], x["responses"]
        if keras.config.backend() == "tensorflow":
            # Tensorflow backend needs uniform ouput types.
            images = tf.convert_to_tensor(images)
        prompts = convert_inputs_to_list_of_tensor_segments(prompts)[0]
        prompts = self.tokenizer(prompts)
        responses = convert_inputs_to_list_of_tensor_segments(responses)[0]
        responses = self.tokenizer(responses)
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
        prompts = convert_inputs_to_list_of_tensor_segments(prompts)[0]
        prompts = self.tokenizer(prompts)
        segments = [prompts]
        if "responses" in x:
            responses = x["responses"]
            responses = convert_inputs_to_list_of_tensor_segments(responses)[0]
            segments.append(self.tokenizer(responses))
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
