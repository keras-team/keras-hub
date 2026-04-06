import re

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_image_converter import (
    Qwen3_5ImageConverter,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


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

    # Special tokens that the KerasHub BPE tokenizer may not encode
    # as single tokens. Map from string → token ID.
    SPECIAL_TOKEN_MAP = {
        "<|im_start|>": 248045,
        "<|im_end|>": 248044,
        "<|vision_start|>": 248053,
        "<|vision_end|>": 248054,
        "<|image_pad|>": 248056,
    }

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

        # Regex pattern for splitting prompts at special token boundaries.
        self._special_token_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in self.SPECIAL_TOKEN_MAP) + ")"
        )

    def _tokenize_with_special_tokens(self, text, num_vision_tokens_per_image):
        """Tokenize text while correctly handling special tokens.

        The KerasHub BPE tokenizer may not encode Qwen3.5's added special
        tokens (``<|image_pad|>``, ``<|vision_start|>``, etc.) as single
        tokens — it can break them into sub-word pieces. This method
        splits the input by known special tokens, tokenizes only the
        text segments, and manually inserts the correct token IDs.

        For ``<|image_pad|>`` tokens, each occurrence is expanded to
        ``N`` copies based on ``num_vision_tokens_per_image``.

        Args:
            text: str. The prompt string.
            num_vision_tokens_per_image: list[int]. Number of vision tokens
                for each image.
        Returns:
            list[int]. The complete token ID sequence.
        """
        parts = self._special_token_pattern.split(text)

        all_ids = []
        img_idx = 0
        for part in parts:
            if part in self.SPECIAL_TOKEN_MAP:
                if part == self.image_token:
                    # Expand image placeholder to N copies.
                    if img_idx < len(num_vision_tokens_per_image):
                        n = num_vision_tokens_per_image[img_idx]
                        img_idx += 1
                    else:
                        n = 1
                    all_ids.extend([self.image_token_id] * n)
                else:
                    all_ids.append(self.SPECIAL_TOKEN_MAP[part])
            elif part:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    def _compute_vision_indices(self, token_ids):
        """Return flat indices where ``token_ids == image_token_id``.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
        Returns:
            int32 tensor ``(total_vision_tokens,)``.
        """
        mask = tf.equal(token_ids, self.image_token_id)
        flat_mask = tf.reshape(mask, [-1])
        return tf.cast(tf.where(flat_mask)[:, 0], "int32")

    def _compute_position_ids(self, token_ids, image_grid_thw):
        """Build 4-channel M-RoPE position IDs matching HF's algorithm.

        For text tokens all 4 channels have the same sequential position.
        For vision tokens channels 1-3 encode (temporal, height, width)
        grid coordinates. Channel 0 mirrors channel 1 (temporal).

        This matches HF's ``get_rope_index`` / ``get_vision_position_ids``:
        - temporal: ``full(n_tokens, start_pos)`` (all same for images)
        - height: ``arange(h_merged).repeat_interleave(w_merged * t)``
        - width: ``arange(w_merged).repeat(h_merged * t)``
        - text_pos advances by ``max(h_merged, w_merged)`` after vision span.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
            image_grid_thw: int32 tensor ``(num_images, 3)`` — ``[T, H, W]``
                per image in patch units (BEFORE spatial merging).
        Returns:
            int32 tensor ``(batch, 4, seq_len)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        if hasattr(image_grid_thw, "numpy"):
            grid_np = ops.convert_to_numpy(image_grid_thw)
        else:
            grid_np = np.array(image_grid_thw)

        batch_size, seq_len = token_ids_np.shape
        merge_size = getattr(self.image_converter, "spatial_merge_size", 2)

        all_pos = np.zeros((batch_size, 4, seq_len), dtype=np.int32)

        for b in range(batch_size):
            ids = token_ids_np[b]
            t_pos = np.zeros(seq_len, dtype=np.int32)
            h_pos = np.zeros(seq_len, dtype=np.int32)
            w_pos = np.zeros(seq_len, dtype=np.int32)

            current_pos = 0
            img_idx = 0
            i = 0
            while i < seq_len:
                if ids[i] == self.image_token_id and img_idx < grid_np.shape[0]:
                    t_grid = int(grid_np[img_idx, 0])
                    h_grid = int(grid_np[img_idx, 1])
                    w_grid = int(grid_np[img_idx, 2])

                    llm_grid_t = t_grid  # temporal merge = 1 for images
                    llm_grid_h = h_grid // merge_size
                    llm_grid_w = w_grid // merge_size
                    n_tokens = llm_grid_t * llm_grid_h * llm_grid_w
                    span_end = min(i + n_tokens, seq_len)

                    # Replicate HF's get_vision_position_ids exactly:
                    #   temporal: all same (current_pos)
                    #   height: repeat_interleave(w * t)
                    #   width: repeat(h * t)
                    for vi in range(span_end - i):
                        t_pos[i + vi] = current_pos
                        row = vi // llm_grid_w  # floor div
                        col = vi % llm_grid_w
                        h_pos[i + vi] = current_pos + (row % llm_grid_h)
                        w_pos[i + vi] = current_pos + col

                    # Advance by max(h_merged, w_merged) — HF convention.
                    current_pos += max(llm_grid_h, llm_grid_w)
                    i = span_end
                    img_idx += 1
                else:
                    t_pos[i] = current_pos
                    h_pos[i] = current_pos
                    w_pos[i] = current_pos
                    current_pos += 1
                    i += 1

            # Channel layout: [text, temporal, height, width].
            # Channel 0 (text) mirrors temporal — for text tokens all
            # channels are identical; for vision tokens the model's
            # attention layer only uses channels 1-3.
            all_pos[b, 0] = t_pos
            all_pos[b, 1] = t_pos
            all_pos[b, 2] = h_pos
            all_pos[b, 3] = w_pos

        return tf.constant(all_pos, dtype="int32")

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

        # Multimodal path.
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts = x["prompts"]

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, 0)

        # 1. Process images to get pixel_values & grid_thw.
        vision_out = self._preprocess_images(images, batched)

        # 2. Compute per-image vision token count from grid_thw.
        grid_thw = vision_out["image_grid_thw"]
        grid_np = (
            grid_thw.numpy()
            if hasattr(grid_thw, "numpy")
            else np.array(grid_thw)
        )
        merge_size = getattr(self.image_converter, "spatial_merge_size", 2)
        num_vision_tokens_per_image = []
        for i in range(grid_np.shape[0]):
            t, h, w = (
                int(grid_np[i, 0]),
                int(grid_np[i, 1]),
                int(grid_np[i, 2]),
            )
            num_vision_tokens_per_image.append(
                t * (h // merge_size) * (w // merge_size)
            )

        # 3. Tokenize with special-token-aware splitting.
        if isinstance(prompts, tf.Tensor):
            prompts_list = [p.numpy().decode("utf-8") for p in prompts]
        elif isinstance(prompts, (list, tuple)):
            prompts_list = [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ]
        else:
            prompts_list = [str(prompts)]

        expanded_sequences = []
        for prompt_str in prompts_list:
            ids = self._tokenize_with_special_tokens(
                prompt_str, num_vision_tokens_per_image
            )
            expanded_sequences.append(ids)

        # 4. Pack to fixed length.
        token_ids_ragged = tf.ragged.constant(expanded_sequences, dtype="int32")
        token_ids, padding_mask = self.packer(
            token_ids_ragged,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        # 5. Compute vision indices & M-RoPE position IDs.
        vision_indices = self._compute_vision_indices(token_ids)
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
        # Normalize to a flat list of individual 3-D images.
        if isinstance(images, (list, tuple)):
            flat_images = []
            for img in images:
                if hasattr(img, "shape") and len(img.shape) == 4:
                    for i in range(img.shape[0]):
                        flat_images.append(img[i])
                else:
                    flat_images.append(img)
        elif hasattr(images, "shape") and len(images.shape) == 4:
            flat_images = [images[i] for i in range(images.shape[0])]
        elif hasattr(images, "shape") and len(images.shape) == 3:
            flat_images = [images]
        else:
            flat_images = [images]

        all_patches = []
        all_grid_thw = []
        for img in flat_images:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            result = self.image_converter(img)
            patches = result["patches"]
            grid_thw = result["grid_thw"]

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

        return {
            "pixel_values": tf.concat(all_patches, axis=0),
            "image_grid_thw": tf.stack(all_grid_thw, axis=0),
        }

    @preprocessing_function
    def generate_postprocess(self, x):
        """Convert integer token output to strings for generation.

        Strips all Qwen3.5 special tokens (vision markers, image pad,
        chat markers) from the output before detokenizing.
        """
        if not self.built:
            self.build(None)

        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        padding_mask = keras.ops.convert_to_numpy(x["padding_mask"])

        # Collect all IDs to strip: base special tokens + vision tokens.
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self.SPECIAL_TOKEN_MAP.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        output = self.tokenizer.detokenize(token_ids)

        # Safety net: strip residual special token strings that may
        # survive if the BPE model encodes them as byte-fallback pieces.
        for tok_str in self.SPECIAL_TOKEN_MAP:
            output = tf.strings.regex_replace(output, re.escape(tok_str), "")
        return output

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
