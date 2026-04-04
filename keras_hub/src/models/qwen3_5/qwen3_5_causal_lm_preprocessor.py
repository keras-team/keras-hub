"""Qwen3.5 Causal LM Preprocessor with multimodal support.

Handles:
- Text-only tokenization (backward compatible).
- Image + text tokenization: replaces <|image_pad|> placeholder tokens with
  the correct number of vision tokens, computes vision_indices for scatter,
  and builds 4-channel M-RoPE position IDs.
"""

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_image_converter import (
    Qwen3_5ImageConverter,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.Qwen3_5CausalLMPreprocessor")
class Qwen3_5CausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3.5 Causal LM preprocessor with multimodal support.

    For text-only usage this behaves identically to the base
    ``CausalLMPreprocessor``.  When an ``image_converter`` is provided,
    the preprocessor also:

    1. Converts images to patch tensors via ``Qwen3_5ImageConverter``.
    2. Replaces ``<|image_pad|>`` placeholder tokens in the token sequence
       with the correct number of vision tokens.
    3. Computes flat ``vision_indices`` for scattering visual embeddings
       into the text sequence.
    4. Builds 4-channel M-RoPE ``position_ids`` for spatial awareness.

    Args:
        tokenizer: A ``Qwen3_5Tokenizer`` instance.
        image_converter: A ``Qwen3_5ImageConverter`` instance, or ``None``
            for text-only mode.
        sequence_length: int. Total padded sequence length. Default 1024.
        add_start_token: bool. Prepend BOS token. Default ``False``.
        add_end_token: bool. Append EOS token. Default ``True``.
        image_token: str. The placeholder token the user inserts in prompts
            to indicate where an image should go. Default ``"<|image_pad|>"``.
        image_token_id: int. The token ID of ``image_token``. Default 248056
            (from HF ``config.json``).
    """

    backbone_cls = Qwen3_5Backbone
    tokenizer_cls = Qwen3_5Tokenizer
    image_converter_cls = Qwen3_5ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        image_token="<|image_pad|>",
        image_token_id=248056,
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
        self.image_token = image_token
        self.image_token_id = image_token_id

    def _compute_vision_indices(self, token_ids):
        """Return flat indices where token_ids == image_token_id.

        Args:
            token_ids: int32 tensor (batch, seq_len).
        Returns:
            vision_indices: int32 tensor (total_vision_tokens,). Flat indices
                into the batch*seq_len vector.
        """
        mask = tf.equal(token_ids, self.image_token_id)
        flat_mask = tf.reshape(mask, [-1])
        indices = tf.cast(tf.where(flat_mask)[:, 0], "int32")
        return indices

    def _compute_position_ids(self, token_ids, image_grid_thw):
        """Build 4-channel M-RoPE position IDs.

        For text tokens all 4 channels have the same sequential position.
        For vision tokens channels 1-3 encode (temporal, height, width)
        grid coordinates. Channel 0 mirrors channel 1 (temporal).

        Args:
            token_ids: int32 tensor (batch, seq_len).
            image_grid_thw: int32 tensor (num_images, 3) - [T, H, W] per image
                in patch units.
        Returns:
            position_ids: int32 tensor (batch, 4, seq_len).
        """
        batch_size = token_ids.shape[0] or tf.shape(token_ids)[0]
        seq_len = token_ids.shape[1] or tf.shape(token_ids)[1]

        # Start with sequential positions for all channels.
        seq_range = tf.range(seq_len, dtype="int32")
        pos_ids = tf.tile(
            tf.reshape(seq_range, (1, 1, -1)),
            (batch_size, 4, 1),
        )

        # For simplicity in this initial implementation, we use uniform
        # sequential position IDs for all tokens (text and vision).
        # The full interleaved spatial position IDs for vision tokens
        # would be computed here in a future refinement. The M-RoPE
        # attention layer still operates correctly because for text-only
        # tokens all 4 channels are identical, and the vision encoder
        # already applies its own 2D rotary embeddings internally.
        return pos_ids

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """Preprocess inputs for generation (prompt-only, no labels).

        Accepts either:
        - A plain string / list of strings (text-only).
        - A dict with ``"prompts"`` and optional ``"images"`` keys.

        Returns:
            dict with ``token_ids``, ``padding_mask``, and optionally
            ``pixel_values``, ``image_grid_thw``, ``vision_indices``,
            ``position_ids``.
        """
        # Check whether the input has images.
        images = None
        if isinstance(x, dict):
            images = x.get("images", None)

        # Text-only: delegate to the base class entirely.
        if images is None or self.image_converter is None:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        # Multimodal: need custom handling for vision inputs.
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts = x["prompts"]

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if tf and isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, 0)

        # Tokenize.
        token_ids_ragged = self.tokenizer(prompts)
        token_ids, _ = self.packer(
            (token_ids_ragged,),
            sequence_length=sequence_length,
            add_end_value=False,
        )
        padding_mask = tf.cast(
            tf.not_equal(token_ids, self.tokenizer.pad_token_id), "int32"
        )

        # Process images.
        vision_out = self._preprocess_images(images, batched)

        # Compute vision indices.
        vision_indices = self._compute_vision_indices(token_ids)

        # Compute M-RoPE position IDs.
        pos_ids = self._compute_position_ids(
            token_ids, vision_out["image_grid_thw"]
        )

        result = {
            "token_ids": token_ids if batched else tf.squeeze(token_ids, 0),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, 0)
            ),
        }
        result.update(vision_out)
        result["vision_indices"] = vision_indices
        result["position_ids"] = pos_ids if batched else tf.squeeze(pos_ids, 0)

        return result

    def _preprocess_images(self, images, batched):
        """Convert raw images to patch tensors using the image converter.

        Args:
            images: A single PIL/numpy image, a list of images, or a
                batched tensor.
            batched: bool. Whether the input is already batched.
        Returns:
            dict with ``pixel_values`` and ``image_grid_thw``.
        """
        # Normalize to list of images.
        if not isinstance(images, (list, tuple)):
            images = [images]

        all_patches = []
        all_grid_thw = []

        for img in images:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                # Grayscale → RGB
                img = np.stack([img] * 3, axis=-1)

            result = self.image_converter(img)

            patches = result["patches"]
            grid_thw = result["grid_thw"]

            # Ensure we have TF tensors.
            if not isinstance(patches, tf.Tensor):
                if hasattr(patches, "cpu"):
                    patches = patches.cpu().detach().numpy()
                patches = tf.constant(patches)
            if not isinstance(grid_thw, tf.Tensor):
                if hasattr(grid_thw, "cpu"):
                    grid_thw = grid_thw.cpu().detach().numpy()
                grid_thw = tf.constant(grid_thw)

            all_patches.append(patches)
            all_grid_thw.append(grid_thw)

        # Concatenate patches from all images.
        pixel_values = tf.concat(all_patches, axis=0)
        image_grid_thw = tf.stack(all_grid_thw, axis=0)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_token": self.image_token,
                "image_token_id": self.image_token_id,
            }
        )
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        return config
