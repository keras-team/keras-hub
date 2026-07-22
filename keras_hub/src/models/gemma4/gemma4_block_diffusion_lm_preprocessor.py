import math
import re

import keras
import numpy as np
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.block_diffusion_lm_preprocessor import (
    BlockDiffusionLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


def _get_num_vision_tokens(
    h, w, patch_size, max_soft_tokens, pooling_kernel_size
):
    total_px = h * w
    max_patches = max_soft_tokens * (pooling_kernel_size**2)
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_h = factor * h
    ideal_w = factor * w
    side_mult = pooling_kernel_size * patch_size

    target_h = int(math.floor(ideal_h / side_mult)) * side_mult
    target_w = int(math.floor(ideal_w / side_mult)) * side_mult

    target_h = max(target_h, side_mult)
    target_w = max(target_w, side_mult)

    n_h = target_h // patch_size
    n_w = target_w // patch_size

    return (n_h * n_w) // (pooling_kernel_size**2)


@keras_hub_export("keras_hub.models.Gemma4BlockDiffusionLMPreprocessor")
class Gemma4BlockDiffusionLMPreprocessor(BlockDiffusionLMPreprocessor):
    """Preprocessing layer for Gemma4 diffusion language model tasks.

    Tokenizes and packs prompt strings, optionally expanding image
    placeholders, then appends `canvas_length` placeholder (pad) tokens
    to form the full model input for the block-diffusion generation loop.

    This layer supports two operational modes:
    - **Text-only**: no converters provided; plain string or dict with
      ``"prompts"`` key accepted.
    - **Image**: pass an ``image_converter``; the prompt may contain
      ``<|image|>`` placeholders that are expanded to per-patch soft tokens.

    The preprocessor accepts input as plain strings **or** as dicts with the
    following keys:

    - ``"prompts"`` (required): string or batch of strings.
    - ``"responses"`` (optional): string or batch of strings used only during
      training (the ``call()`` path).
    - ``"images"`` (optional): image tensor(s) matching ``<|image|>``
      occurrences in the prompts.

    During generation (``generate_preprocess()``), the packed prompt tokens are
    followed by ``canvas_length`` mask tokens.  The model denoises these
    positions iteratively.

    Args:
        tokenizer: A `keras_hub.models.Gemma4Tokenizer` instance.
        image_converter: A `keras_hub.layers.Gemma4ImageConverter` instance.
            Defaults to `None`.
        sequence_length: int. Maximum prompt sequence length. Defaults to
            `256`.
        canvas_length: int. Number of canvas tokens appended after the packed
            prompt during generation preprocessing. Defaults to `256`.
        add_start_token: bool. Whether to prepend the BOS token. Defaults to
            `True`.
        add_end_token: bool. Whether to append the EOS token after the prompt.
            Defaults to `True`.
        max_images_per_prompt: int. Maximum number of images per sample.
            Defaults to `2`.
        num_vision_tokens_per_image: int. Vision placeholder tokens per image.
            Defaults to `280`.
    """

    backbone_cls = Gemma4Backbone
    tokenizer_cls = Gemma4Tokenizer
    image_converter_cls = Gemma4ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        canvas_length=256,
        add_start_token=True,
        add_end_token=True,
        max_images_per_prompt=2,
        num_vision_tokens_per_image=280,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            canvas_length=canvas_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )

        if (
            image_converter is not None
            and sequence_length
            <= max_images_per_prompt * num_vision_tokens_per_image
        ):
            raise ValueError(
                "`sequence_length` should be greater than "
                "`max_images_per_prompt * num_vision_tokens_per_image`."
            )

        self.image_converter = image_converter
        self.max_images_per_prompt = max_images_per_prompt
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

        self.text_only_model = self.image_converter is None

        if self.image_converter is None:
            self.image_placeholder = None
            self.start_of_image_token = None
            self.end_of_image_token = None
        else:
            self.image_placeholder = "<|image|>"
            self.start_of_image_token = "<|image>"
            self.end_of_image_token = "<image|>"

    def build(self, input_shape):
        # Use MultiSegmentPacker so training supports (prompt, response) pairs.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _get_vision_indices(self, vision_mask, max_tokens=None):
        batch_size, sequence_length = vision_mask.shape

        vision_mask_flattened = tf.reshape(vision_mask, [-1])
        vision_indices = tf.where(vision_mask_flattened)[..., 0]
        vision_indices = tf.cast(vision_indices, dtype=tf.int32)

        row_lengths = tf.math.reduce_sum(
            tf.cast(vision_mask, dtype=vision_indices.dtype), axis=1
        )
        batched_vision_indices = tf.RaggedTensor.from_row_lengths(
            values=vision_indices,
            row_lengths=row_lengths,
        )

        to_subtract = tf.math.scalar_mul(
            scalar=tf.cast(sequence_length, dtype=tf.int32),
            x=tf.range(
                start=0,
                limit=tf.shape(vision_mask)[0],
                dtype=tf.int32,
            ),
        )
        batched_vision_indices = tf.math.subtract(
            batched_vision_indices,
            tf.expand_dims(to_subtract, axis=-1),
        )

        pad_shape = [None, max_tokens] if max_tokens is not None else None
        batched_vision_indices = batched_vision_indices.to_tensor(
            default_value=0,
            shape=pad_shape,
        )
        return batched_vision_indices

    def _preprocess_images(self, images, batched):
        if isinstance(images, np.ndarray):
            images = tf.convert_to_tensor(images)
        elif isinstance(images, list):
            images = tf.ragged.constant(images)
        elif not isinstance(images, (tf.Tensor, tf.RaggedTensor)):
            images = tf.convert_to_tensor(images)

        if isinstance(images, tf.RaggedTensor):
            if not batched:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 4:
                images = tf.expand_dims(images, axis=1)
            images = images.to_tensor(
                shape=[None, self.max_images_per_prompt, None, None, 3],
                default_value=0,
            )
        elif isinstance(images, tf.Tensor):
            if not batched:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 3:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 4:
                images = tf.expand_dims(images, axis=1)
        else:
            raise ValueError(
                "`images` should be a list, ragged tensor, or dense tensor."
                f" Received: `type(images)` = {type(images)}"
            )

        original_images_shape = tf.shape(images)
        images = tf.reshape(
            images,
            [
                -1,
                original_images_shape[-3],
                original_images_shape[-2],
                original_images_shape[-1],
            ],
        )
        images_dict = self.image_converter.call(images)
        pixel_values = images_dict["pixel_values"]
        pixel_position_ids = images_dict["pixel_position_ids"]

        if keras.config.backend() == "torch":
            if not isinstance(pixel_values, tf.Tensor):
                pixel_values = pixel_values.cpu()
            if not isinstance(pixel_position_ids, tf.Tensor):
                pixel_position_ids = pixel_position_ids.cpu()

        pixel_values = tf.reshape(
            pixel_values,
            [
                original_images_shape[0],
                original_images_shape[1],
                -1,
                self.image_converter.patch_size**2 * 3,
            ],
        )
        pixel_position_ids = tf.reshape(
            pixel_position_ids,
            [
                original_images_shape[0],
                original_images_shape[1],
                -1,
                2,
            ],
        )
        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

    def _build_multimodal_output(
        self,
        token_ids,
        padding_mask,
        vision_mask,
        pixel_values,
        pixel_position_ids,
        batched,
        canvas_tokens=None,
        canvas_mask=None,
    ):
        """Assemble the output dict from processed tensors.

        When ``canvas_tokens`` / ``canvas_mask`` are provided (generation path),
        they are appended to ``token_ids`` / ``padding_mask`` before building
        ``position_ids``.
        """
        if canvas_tokens is not None:
            token_ids = tf.concat([token_ids, canvas_tokens], axis=1)
            padding_mask = tf.concat([padding_mask, canvas_mask], axis=1)

        batch_size = tf.shape(token_ids)[0]
        seq_len = tf.shape(token_ids)[1]
        position_ids = tf.range(seq_len, dtype=tf.int32)
        position_ids = tf.expand_dims(position_ids, axis=0)
        position_ids = tf.tile(position_ids, [batch_size, 1])

        if self.text_only_model:
            vision_indices = tf.ones([batch_size, 0], dtype=tf.int32)
        else:
            if self.image_converter is not None:
                max_vision_tokens = (
                    self.max_images_per_prompt
                    * self.num_vision_tokens_per_image
                )
            else:
                max_vision_tokens = None
            vision_indices = self._get_vision_indices(
                vision_mask=vision_mask, max_tokens=max_vision_tokens
            )

        if pixel_values is None:
            patch_dim = (
                3 * self.image_converter.patch_size**2
                if self.image_converter is not None
                else 48
            )
            pixel_values = tf.zeros(
                (batch_size, 0, 1, patch_dim), dtype="float32"
            )
        if pixel_position_ids is None:
            pixel_position_ids = tf.zeros((batch_size, 0, 1, 2), dtype="int32")

        x = {
            "token_ids": (
                token_ids if batched else tf.squeeze(token_ids, axis=0)
            ),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, axis=0)
            ),
            "position_ids": (
                position_ids if batched else tf.squeeze(position_ids, axis=0)
            ),
            "pixel_values": (
                pixel_values if batched else tf.squeeze(pixel_values, axis=0)
            ),
            "pixel_position_ids": (
                pixel_position_ids
                if batched
                else tf.squeeze(pixel_position_ids, axis=0)
            ),
            "vision_indices": (
                vision_indices
                if batched
                else tf.squeeze(vision_indices, axis=0)
            ),
            "vision_mask": (
                vision_mask if batched else tf.squeeze(vision_mask, axis=0)
            ),
        }

        return x

    def _expand_and_tokenize_prompts(self, prompts, batched):
        """Expand image placeholders in prompts and return token IDs.

        Returns:
            token_ids_ragged
        """
        if self.image_converter is not None:
            num_tokens = self.num_vision_tokens_per_image
            img_pattern = re.escape(self.image_placeholder)
            prompts = tf.strings.regex_replace(
                prompts,
                img_pattern,
                f"{self.start_of_image_token}"
                + self.image_placeholder * num_tokens
                + f"{self.end_of_image_token}",
            )

        return self.tokenizer(prompts)

    def _resolve_vision(
        self,
        token_ids,
        images,
        pixel_values,
        pixel_position_ids,
        batch_size,
        batched,
    ):
        """Return (pixel_values, pixel_position_ids, vision_mask)."""
        if images is not None and self.image_converter is not None:
            vd = self._preprocess_images(images, batched)
            pixel_values = vd["pixel_values"]
            pixel_position_ids = vd["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        elif pixel_values is not None:
            pixel_values = (
                pixel_values if batched else tf.expand_dims(pixel_values, 0)
            )
            pixel_position_ids = (
                pixel_position_ids
                if batched
                else tf.expand_dims(pixel_position_ids, 0)
            )
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        else:
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
                pixel_values = tf.ones(
                    [batch_size, 0, 0, patch_dim], dtype="float32"
                )
                pixel_position_ids = tf.zeros(
                    [batch_size, 0, 0, 2], dtype="int32"
                )
            else:
                pixel_values = None
                pixel_position_ids = None
            vision_mask = tf.zeros_like(token_ids, dtype=bool)

        return pixel_values, pixel_position_ids, vision_mask

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length

        # Accept plain strings for backward compatibility.
        if not isinstance(x, dict):
            prompts = x
            responses = None
            images = None
            pixel_values = pixel_position_ids = None
        else:
            prompts = x["prompts"]
            responses = x.get("responses", None)
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)

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

        prompts_tok = self._expand_and_tokenize_prompts(prompts, batched)

        if responses is not None:
            responses_tok = self.tokenizer(responses)
            segments = (prompts_tok, responses_tok)
        else:
            segments = (prompts_tok,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Text-only shortcut — emits the same keys as `Gemma4BlockDiffusion
        # LMPreprocessor`'s multimodal branch so the LM can consume
        # `backbone.input` directly.
        if self.text_only_model:
            label_ids = token_ids[..., 1:]
            sw = (
                response_mask[..., 1:]
                if responses is not None
                else padding_mask[..., 1:]
            )
            trimmed_token_ids = token_ids[..., :-1]
            trimmed_padding_mask = padding_mask[..., :-1]
            batch_size = tf.shape(trimmed_token_ids)[0]
            seq_len = tf.shape(trimmed_token_ids)[1]
            position_ids = tf.range(seq_len, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, axis=0)
            position_ids = tf.tile(position_ids, [batch_size, 1])
            out_x = {
                "token_ids": trimmed_token_ids,
                "padding_mask": trimmed_padding_mask,
                "position_ids": position_ids,
            }
            if not batched:
                out_x["token_ids"] = tf.squeeze(out_x["token_ids"], axis=0)
                out_x["padding_mask"] = tf.squeeze(
                    out_x["padding_mask"], axis=0
                )
                out_x["position_ids"] = tf.squeeze(
                    out_x["position_ids"], axis=0
                )
                label_ids = tf.squeeze(label_ids, axis=0)
                sw = tf.squeeze(sw, axis=0)
            return keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)

        batch_size = tf.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids[..., :-1],
            images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Build labels before trimming.
        label_ids = token_ids[..., 1:]
        sw = (
            response_mask[..., 1:]
            if responses is not None
            else padding_mask[..., 1:]
        )
        token_ids = token_ids[..., :-1]
        padding_mask = padding_mask[..., :-1]

        out_x = self._build_multimodal_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            batched=batched,
        )

        if not batched:
            label_ids = tf.squeeze(label_ids, axis=0)
            sw = tf.squeeze(sw, axis=0)

        return keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """Convert prompt inputs to model-ready tensors for generation.

        Expands image placeholders, tokenizes, packs to ``sequence_length``,
        then appends ``canvas_length`` mask tokens for the denoising loop.

        Args:
            x: A string, batch of strings, or a dict with key ``"prompts"``
                and optionally ``"images"``, ``"pixel_values"``,
                ``"pixel_position_ids"``.
            sequence_length: Optional int. Prompt sequence length. Defaults to
                ``self.sequence_length``.

        Returns:
            A dict with ``"token_ids"``, ``"padding_mask"``, and multimodal
            fields when converters are configured.
        """
        if not self.built:
            self.build(None)

        seq_len = sequence_length or self.sequence_length

        if isinstance(x, dict):
            prompts = x["prompts"]
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)
        else:
            prompts = x
            images = None
            pixel_values = pixel_position_ids = None

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)

        if self.text_only_model and (
            pixel_values is not None or images is not None
        ):
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`images`/`pixel_values` is not `None`."
            )

        prompts_tok = self._expand_and_tokenize_prompts(prompts, batched)

        token_ids, segment_ids = self.packer(
            (prompts_tok,),
            sequence_length=seq_len,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Text-only: append canvas and emit position_ids alongside the
        # standard token / mask fields so the LM's functional graph can
        # consume the dict directly.
        if self.text_only_model:
            batch_size = tf.shape(token_ids)[0]
            mask_id = self.tokenizer.pad_token_id
            canvas_tokens = tf.fill([batch_size, self.canvas_length], mask_id)
            canvas_tokens = tf.cast(canvas_tokens, token_ids.dtype)
            canvas_mask = tf.zeros(
                [batch_size, self.canvas_length], dtype=padding_mask.dtype
            )
            token_ids = tf.concat([token_ids, canvas_tokens], axis=1)
            padding_mask = tf.concat([padding_mask, canvas_mask], axis=1)
            seq_len = tf.shape(token_ids)[1]
            position_ids = tf.range(seq_len, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, axis=0)
            position_ids = tf.tile(position_ids, [batch_size, 1])
            if not batched:
                token_ids = tf.squeeze(token_ids, axis=0)
                padding_mask = tf.squeeze(padding_mask, axis=0)
                position_ids = tf.squeeze(position_ids, axis=0)
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "position_ids": position_ids,
            }

        batch_size = tf.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids,
            images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Canvas tokens (mask/pad token_id) are appended in
        # _build_multimodal_output.
        mask_id = self.tokenizer.pad_token_id
        canvas_tokens = tf.fill([batch_size, self.canvas_length], mask_id)
        canvas_tokens = tf.cast(canvas_tokens, token_ids.dtype)
        canvas_mask = tf.zeros(
            [batch_size, self.canvas_length], dtype=padding_mask.dtype
        )

        return self._build_multimodal_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            batched=batched,
            canvas_tokens=canvas_tokens,
            canvas_mask=canvas_mask,
        )

    @preprocessing_function
    def generate_postprocess(self, x):
        """Convert denoised integer token IDs back to strings.

        Strips vision soft tokens in addition to the standard special
        tokens before detokenization.

        Args:
            x: A dict with ``"token_ids"`` and ``"padding_mask"`` (or an int
               tensor of shape ``(B, canvas_length)`` for the raw canvas).

        Returns:
            String or list of strings.
        """
        if not self.built:
            self.build(None)

        if isinstance(x, dict):
            token_ids = keras.ops.convert_to_numpy(x["token_ids"])
            padding_mask = keras.ops.convert_to_numpy(x["padding_mask"])
        else:
            token_ids = keras.ops.convert_to_numpy(x).astype("int32")
            padding_mask = (token_ids != self.tokenizer.pad_token_id).astype(
                bool
            )

        ids_to_strip = list(getattr(self.tokenizer, "special_token_ids", []))

        if self.image_converter is not None:
            soi_id = getattr(self.tokenizer, "start_of_image_token_id", None)
            if soi_id is not None and soi_id in ids_to_strip:
                ids_to_strip.remove(soi_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        output = self.tokenizer.detokenize(token_ids)

        return output

    @property
    def max_images_per_prompt(self):
        return self._max_images_per_prompt

    @max_images_per_prompt.setter
    def max_images_per_prompt(self, value):
        self._max_images_per_prompt = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": None
                if self.image_converter is None
                else keras.layers.serialize(self.image_converter),
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
                "max_images_per_prompt": self.max_images_per_prompt,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.update(
            {
                "image_converter": None
                if config.get("image_converter") is None
                else keras.layers.deserialize(config["image_converter"]),
            }
        )
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.saving.deserialize_keras_object(
                config["tokenizer"]
            )
        return cls(**config)
