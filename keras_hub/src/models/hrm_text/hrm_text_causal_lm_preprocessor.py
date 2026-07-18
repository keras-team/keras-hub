"""Causal language-model preprocessing for HRM-Text."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.HrmTextCausalLMPreprocessor")
class HrmTextCausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocesses causal and PrefixLM data for `HrmTextCausalLM`.

    A plain string is treated as causal language-model text. For PrefixLM
    training, pass a dictionary containing ``prefix`` and ``response``. Only
    response-token labels receive training weight; prefix tokens can attend to
    one another bidirectionally.
    """

    backbone_cls = HrmTextBackbone
    tokenizer_cls = HrmTextTokenizer

    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            sep_value=self.tokenizer.end_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            truncate="waterfall",
        )
        self.built = True

    def _pack(self, segments, sequence_length, add_end_value):
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=add_end_value,
        )
        padding_mask = ops.cast(
            token_ids != self.tokenizer.pad_token_id, "int32"
        )
        return token_ids, padding_mask, segment_ids

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        if isinstance(x, dict):
            segments = (
                self.tokenizer(x["prefix"]),
                self.tokenizer(x["response"]),
            )
            prefix_lm = True
        else:
            segments = self.tokenizer(x)
            prefix_lm = False
        token_ids, padding_mask, segment_ids = self._pack(
            segments,
            sequence_length + 1,
            add_end_value=self.add_end_token,
        )
        if prefix_lm:
            token_type_ids = ops.cast(segment_ids == 0, "int32") * padding_mask
        else:
            token_type_ids = ops.zeros_like(padding_mask)
        inputs = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "token_type_ids": token_type_ids[..., :-1],
        }
        labels = token_ids[..., 1:]
        weights = padding_mask[..., 1:]
        if prefix_lm:
            weights = weights * ops.cast(segment_ids[..., 1:] == 1, "int32")
        return keras.utils.pack_x_y_sample_weight(inputs, labels, weights)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        return self._call_python(
            x, y=y, sample_weight=sample_weight, sequence_length=sequence_length
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask, _ = self._pack(
            self.tokenizer(x),
            sequence_length=sequence_length,
            add_end_value=False,
        )
        return {
            "token_ids": token_ids,
            "padding_mask": ops.cast(padding_mask, "int32"),
            "token_type_ids": ops.cast(padding_mask, "int32"),
        }

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        return self._generate_preprocess_python(x, sequence_length)

    def generate_preprocess(self, x, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )
