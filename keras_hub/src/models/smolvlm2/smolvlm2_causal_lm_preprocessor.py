import re

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function


def _get_image_prompt_string(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_image_token,
):
    """Build the expanded image prompt string.

    Replicates HF's ``_prompt_split_image`` / ``_prompt_single_image``.

    When ``image_rows == 0 and image_cols == 0``, the image was not split
    and a single global-image block is produced.  Otherwise, each
    ``(row, col)`` patch gets its own block plus a trailing global view.

    Args:
        image_seq_len: int. Number of ``<image>`` tokens per sub-image.
        image_rows: int. Number of row splits (0 if unsplit).
        image_cols: int. Number of column splits (0 if unsplit).
        fake_token_around_image: str.
        image_token: str.
        global_image_token: str.
    Returns:
        str. The expanded prompt fragment.
    """
    if image_rows == 0 and image_cols == 0:
        # Single unsplit image.
        return (
            f"{fake_token_around_image}"
            f"{global_image_token}"
            f"{image_token * image_seq_len}"
            f"{fake_token_around_image}"
        )

    # Split image: row×col patches + 1 global.
    text = ""
    for r in range(image_rows):
        for c in range(image_cols):
            text += (
                f"{fake_token_around_image}"
                f"<row_{r + 1}_col_{c + 1}>"
                f"{image_token * image_seq_len}"
            )
        text += "\n"
    text += (
        f"\n{fake_token_around_image}"
        f"{global_image_token}"
        f"{image_token * image_seq_len}"
        f"{fake_token_around_image}"
    )
    return text


@keras_hub_export("keras_hub.models.SmolVLM2CausalLMPreprocessor")
class SmolVLM2CausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocessor for SmolVLM2 causal language models.

    This preprocessing layer handles tokenization and image processing
    for the SmolVLM2 model. It tokenizes text inputs, processes images
    via the image converter, and packs them into the format expected by
    ``SmolVLM2CausalLM``.

    For multimodal inputs, prompts should contain a single ``<image>``
    placeholder per image. The preprocessor expands each ``<image>``
    into the full HF-style token sequence:

        ``<fake_token_around_image><row_R_col_C><image>×N ... ``
        ``<global-img><image>×N<fake_token_around_image>``

    The expansion count ``N`` (``image_seq_len``) is computed as::

        N = (max_image_size / patch_size)² / scale_factor²

    For text-only inputs, no ``pixel_values`` or ``vision_indices``
    are returned, following the Qwen3.5 pattern.

    Args:
        tokenizer: A ``SmolVLM2Tokenizer`` instance.
        image_converter: A ``SmolVLM2ImageConverter`` instance or ``None``.
        sequence_length: int. Maximum sequence length. Default ``1024``.
        add_start_token: bool. Whether to prepend BOS. Default ``False``
            (matching HF's ``add_bos_token``).
        add_end_token: bool. Whether to append EOS. Default ``True``.
        image_seq_len: int. Number of ``<image>`` tokens per sub-image
            crop.  Default ``64`` for the 256M preset
            (``(512/16)² / 4² = 64``).
    """

    backbone_cls = SmolVLM2Backbone
    tokenizer_cls = SmolVLM2Tokenizer
    image_converter_cls = SmolVLM2ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        image_seq_len=64,
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
        self.image_seq_len = image_seq_len

    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    # ------------------------------------------------------------------
    # Special-token-aware tokenization
    # ------------------------------------------------------------------
    def _build_special_token_map(self):
        """Return {token_string: token_id} for all registered specials."""
        special_map = {}
        for attr in [
            "start_token",
            "end_token",
            "image_token",
            "end_of_utterance_token",
            "fake_image_token",
            "global_image_token",
        ]:
            tok_str = getattr(self.tokenizer, attr, None)
            tok_id = getattr(self.tokenizer, f"{attr}_id", None)
            if tok_str is not None and tok_id is not None:
                special_map[tok_str] = tok_id
        return special_map

    def _tokenize_with_special_tokens(self, text, special_map=None):
        """Tokenize text while preserving special tokens as single IDs.

        The KerasHub BPE tokenizer may break added special tokens into
        sub-word pieces. This method splits the input by known specials,
        tokenizes only the text segments, and manually inserts the
        correct token IDs.

        Additionally, ``<row_R_col_C>`` positional tokens are recognised
        and mapped to their vocabulary IDs.

        Args:
            text: str. The fully-expanded prompt string.
            special_map: dict or None. ``{token_str: token_id}``.
        Returns:
            list[int]. The complete token ID sequence.
        """
        if special_map is None:
            special_map = self._build_special_token_map()

        # Add row/col tokens dynamically.
        row_col_pattern = re.compile(r"<row_\d+_col_\d+>")
        for match in row_col_pattern.finditer(text):
            token = match.group()
            if token not in special_map:
                # Look up in vocabulary.
                vocab = self.tokenizer.get_vocabulary()
                if isinstance(vocab, dict):
                    tid = vocab.get(token, None)
                else:
                    # vocab is a list — build lookup.
                    try:
                        tid = vocab.index(token)
                    except (ValueError, AttributeError):
                        tid = None
                if tid is not None:
                    special_map[token] = tid

        # Build regex for splitting.
        escaped = [re.escape(t) for t in special_map]
        # Also match <row_R_col_C> generically.
        pattern = re.compile(
            "(" + "|".join(escaped) + "|" + r"<row_\d+_col_\d+>" + ")"
        )

        parts = pattern.split(text)
        all_ids = []
        for part in parts:
            if part in special_map:
                all_ids.append(special_map[part])
            elif row_col_pattern.fullmatch(part):
                # Unknown row/col token — skip (shouldn't happen).
                pass
            elif part:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    # ------------------------------------------------------------------
    # Prompt extraction helper
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_prompt_string(prompts):
        """Extract a Python str from various input types."""
        if isinstance(prompts, (list, tuple)):
            val = prompts[0]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            if hasattr(val, "numpy"):
                val = val.numpy()
                if isinstance(val, bytes):
                    return val.decode("utf-8")
            return str(val)
        if isinstance(prompts, str):
            return prompts
        if isinstance(prompts, bytes):
            return prompts.decode("utf-8")
        # tf.Tensor, np.ndarray, or np scalar.
        val = prompts
        if hasattr(val, "numpy"):
            val = val.numpy()
        if isinstance(val, np.ndarray):
            val = val.flat[0] if val.ndim > 0 else val.item()
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return str(val)

    # ------------------------------------------------------------------
    # Vision indices
    # ------------------------------------------------------------------
    def _compute_vision_indices(self, token_ids):
        """Return flat indices where token_ids == image_token_id."""
        token_ids_np = (
            token_ids.numpy()
            if hasattr(token_ids, "numpy")
            else np.array(token_ids)
        )
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None:
            return tf.zeros((0,), dtype=tf.int32)

        mask = token_ids_np.reshape(-1) == image_token_id
        indices = np.where(mask)[0].astype(np.int32)
        return tf.constant(indices)

    # ------------------------------------------------------------------
    # call (training)
    # ------------------------------------------------------------------
    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Handle both dict and string inputs.
        if isinstance(x, dict):
            images = x.get("images", None)
            prompts = x["prompts"]
            responses = x["responses"]
        else:
            images = None
            prompts = x
            responses = x

        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)
        if images is not None and self.image_converter:
            images = self.image_converter(images)

        # Pad with one extra token for truncation below.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        batch_size = tf.shape(token_ids)[0]

        out = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }

        # Always include vision keys — the backbone functional graph
        # requires all 4 inputs.  For text-only, use dummy tensors.
        if images is not None and self.image_converter:
            if isinstance(images, dict):
                out["pixel_values"] = images["pixel_values"]
            else:
                out["pixel_values"] = images
        else:
            # Dummy pixel values sized to image_size (from backbone).
            dummy_size = 32  # will be overridden by real preset
            out["pixel_values"] = tf.zeros(
                (batch_size, dummy_size, dummy_size, 3),
                dtype="float32",
            )
        out["vision_indices"] = tf.zeros((batch_size, 0), dtype=tf.int32)

        y = token_ids[..., 1:]
        sample_weight = response_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(out, y, sample_weight)

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _preprocess_images(self, images):
        """Normalize image inputs and process through the converter.

        Handles lists, batched 4-D arrays, and single 3-D images.
        Currently supports one image per prompt.

        Args:
            images: A single image (3-D), a batch (4-D), or a list.
        Returns:
            dict with ``pixel_values`` (N, H, W, 3), ``rows``, ``cols``.
        """
        # Flatten to a single 3-D image.
        if isinstance(images, (list, tuple)):
            img = images[0]
            if hasattr(img, "shape") and len(img.shape) == 4:
                img = img[0]
        elif hasattr(images, "shape") and len(images.shape) == 4:
            img = images[0]
        elif hasattr(images, "shape") and len(images.shape) == 3:
            img = images
        else:
            img = images

        if isinstance(img, np.ndarray) and img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if self.image_converter is not None:
            result = self.image_converter(img)
            if isinstance(result, dict):
                pixel_values = result["pixel_values"]
                rows = result.get("rows", 0)
                cols = result.get("cols", 0)
            else:
                pixel_values = result
                rows = 0
                cols = 0
        else:
            pixel_values = np.array(img, dtype="float32")
            if pixel_values.ndim == 3:
                pixel_values = np.expand_dims(pixel_values, 0)
            rows = 0
            cols = 0

        # Ensure numpy array for downstream use.
        if not isinstance(pixel_values, np.ndarray):
            pixel_values = ops.convert_to_numpy(pixel_values)

        return {"pixel_values": pixel_values, "rows": rows, "cols": cols}

    # ------------------------------------------------------------------
    # generate_preprocess
    # ------------------------------------------------------------------
    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert inputs to integer token IDs for generation.

        For text-only inputs, returns only ``token_ids`` and
        ``padding_mask``.  For multimodal inputs, processes images
        through the image converter, expands ``<image>`` tokens using
        HF's prompt format, and returns ``pixel_values`` and
        ``vision_indices``.

        Multimodal prompts must contain ``<image>`` placeholder tokens.
        Each ``<image>`` is expanded to the full sub-image token
        sequence matching HuggingFace's ``SmolVLMProcessor``.
        """
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length

        # Handle both dict and string inputs.
        if isinstance(x, dict):
            images = x.get("images", None)
            prompts = x["prompts"]
        else:
            images = None
            prompts = x

        # ------ Text-only path ------
        if images is None:
            prompts = self.tokenizer(prompts)
            token_ids, segment_ids = self.packer(
                (prompts,),
                sequence_length=sequence_length,
                add_start_value=self.add_start_token,
                add_end_value=False,
            )
            padding_mask = token_ids != self.tokenizer.pad_token_id
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }

        # ------ Multimodal path ------
        # Process image through the converter (Qwen3.5 pattern).
        image_output = self._preprocess_images(images)
        pixel_values = image_output["pixel_values"]
        image_rows = int(image_output["rows"])
        image_cols = int(image_output["cols"])

        # Extract prompt string.
        prompt_str = self._extract_prompt_string(prompts)

        # Get special token strings.
        image_token_str = getattr(self.tokenizer, "image_token", "<image>")
        fake_image_str = getattr(
            self.tokenizer,
            "fake_image_token",
            "<fake_token_around_image>",
        )
        global_image_str = getattr(
            self.tokenizer, "global_image_token", "<global-img>"
        )

        # Expand each <image> placeholder with HF's format.
        image_prompt = _get_image_prompt_string(
            image_seq_len=self.image_seq_len,
            image_rows=image_rows,
            image_cols=image_cols,
            fake_token_around_image=fake_image_str,
            image_token=image_token_str,
            global_image_token=global_image_str,
        )
        # Replace the first <image> in the prompt.
        expanded_prompt = prompt_str.replace(image_token_str, image_prompt, 1)

        # Tokenize with special token handling.
        special_map = self._build_special_token_map()
        token_ids_list = self._tokenize_with_special_tokens(
            expanded_prompt, special_map
        )

        # Pack to fixed length.
        token_ids_tensor = tf.ragged.constant([token_ids_list], dtype="int32")
        token_ids, segment_ids = self.packer(
            (token_ids_tensor,),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Compute vision_indices.
        vision_indices = self._compute_vision_indices(token_ids)

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "vision_indices": vision_indices,
        }
