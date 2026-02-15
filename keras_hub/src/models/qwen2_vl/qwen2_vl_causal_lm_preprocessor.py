"""Qwen2-VL Causal LM Preprocessor.

Handles tokenization, image preprocessing, vision token placement,
and M-RoPE position ID computation for Qwen2-VL.
"""

import keras
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.Qwen2VLCausalLMPreprocessor")
class Qwen2VLCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen2-VL Causal LM preprocessor.

    This preprocessing layer is meant for use with
    ``keras_hub.models.Qwen2VLCausalLM``. It takes in batches of strings
    (and optionally images), and returns outputs in a
    ``(x, y, sample_weight)`` format, where the ``y`` label is the next
    token id in the ``x`` sequence. ``sample_weight`` is 0 for "prompt"
    tokens, and 1 for "response" tokens, so that the loss is computed only
    on the "response" tokens.

    For use with generation, the layer also exposes two methods
    ``generate_preprocess()`` and ``generate_postprocess()``.

    Args:
        tokenizer: A ``keras_hub.models.Qwen2VLTokenizer`` instance.
        image_converter: A ``keras_hub.layers.Qwen2VLImageConverter`` instance.
            Defaults to ``None``.
        sequence_length: int. The length of the packed inputs.
            Defaults to 1024.
        add_start_token: bool. Whether to prepend the start token.
            Defaults to ``False`` (Qwen models do not use a start token).
        add_end_token: bool. Whether to append the end token.
            Defaults to ``True``.
    """

    backbone_cls = Qwen2VLBackbone
    tokenizer_cls = Qwen2VLTokenizer
    image_converter_cls = Qwen2VLImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=False,
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
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.end_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _compute_mrope_position_ids(self, token_ids, padding_mask):
        """Compute M-RoPE position IDs for text-only input.

        For text-only mode, all three components (temporal, height, width)
        are identical and equal to sequential position indices.

        Args:
            token_ids: Tensor of shape ``(batch, seq_len)``.
            padding_mask: Tensor of shape ``(batch, seq_len)``.

        Returns:
            Tensor of shape ``(batch, seq_len, 3)``.
        """
        seq_len = tf.shape(token_ids)[1]
        positions = tf.range(seq_len, dtype=tf.int32)
        # Broadcast to (batch, seq_len)
        batch_size = tf.shape(token_ids)[0]
        positions = tf.broadcast_to(positions, (batch_size, seq_len))
        # Mask out padding positions
        positions = positions * tf.cast(padding_mask, tf.int32)
        # Stack 3 identical copies for text-only M-RoPE
        mrope_position_ids = tf.stack(
            [positions, positions, positions], axis=-1
        )
        return mrope_position_ids

    def _format_output(
        self,
        token_ids,
        padding_mask,
        mrope_position_ids,
        response_mask,
        return_labels=False,
        batched=False,
    ):
        """Format output dictionary, optionally computing labels."""
        if return_labels:
            y = token_ids[..., 1:]
            sample_weight = response_mask[..., 1:]
            token_ids = token_ids[..., :-1]
            padding_mask = padding_mask[..., :-1]
            mrope_position_ids = mrope_position_ids[..., :-1, :]

        x = {
            "token_ids": (
                token_ids if batched else tf.squeeze(token_ids, axis=0)
            ),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, axis=0)
            ),
            "mrope_position_ids": (
                mrope_position_ids
                if batched
                else tf.squeeze(mrope_position_ids, axis=0)
            ),
        }

        if return_labels:
            if not batched:
                y = tf.squeeze(y, axis=0)
                sample_weight = tf.squeeze(sample_weight, axis=0)
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        return x

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Extract text.
        prompts, responses = x["prompts"], x["responses"]

        # Handle batching.
        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            responses = tf.expand_dims(responses, axis=0)

        # Tokenize.
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # Pack.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Compute M-RoPE position IDs.
        mrope_position_ids = self._compute_mrope_position_ids(
            token_ids, padding_mask
        )

        return self._format_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            mrope_position_ids=mrope_position_ids,
            response_mask=response_mask,
            return_labels=True,
            batched=batched,
        )

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in
        strings or tensor strings, tokenizes and packs the input, and
        computes a padding mask and M-RoPE position IDs.

        Unlike calling the layer for training, this method does not
        compute labels and will never append a ``tokenizer.end_token_id``
        to the end of the sequence.
        """
        if not self.built:
            self.build(None)

        # Extract inputs.
        if isinstance(x, dict):
            prompts = x["prompts"]
            responses = x.get("responses", None)
        else:
            prompts = x
            responses = None

        # Handle batching.
        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if responses is not None:
                responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            if responses is not None:
                responses = tf.expand_dims(responses, axis=0)

        # Tokenize.
        prompts = self.tokenizer(prompts)
        if responses is not None:
            responses = self.tokenizer(responses)
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        # Pack (no end token for generation).
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Compute M-RoPE position IDs.
        mrope_position_ids = self._compute_mrope_position_ids(
            token_ids, padding_mask
        )

        return self._format_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            mrope_position_ids=mrope_position_ids,
            response_mask=segment_ids == 1,
            return_labels=False,
            batched=batched,
        )

    def generate_postprocess(self, x):
        """Convert integer token output to strings for generation."""
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        # Strip padding and special tokens.
        token_ids = tf.ragged.boolean_mask(token_ids, padding_mask)
        return self.tokenizer.detokenize(token_ids)

    def get_config(self):
        config = super().get_config()
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        return config
