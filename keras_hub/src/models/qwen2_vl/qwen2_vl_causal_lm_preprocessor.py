import re

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.utils.tensor_utils import assert_tf_installed
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Qwen2VLCausalLMPreprocessor")
class Qwen2VLCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen2-VL Causal LM preprocessor with multimodal support.

    For text-only usage this behaves identically to the base
    ``CausalLMPreprocessor``. When an ``image_converter`` is provided, the
    preprocessor also:

    1. Converts images/videos to patch tensors via
       ``Qwen2VLImageConverter``.
    2. Replaces each ``<|image_pad|>``/``<|video_pad|>`` marker in the prompt
       with the correct number of vision placeholder tokens.
    3. Computes flat ``vision_indices`` for scattering visual embeddings
       into the text sequence.
    4. Builds 3-channel M-RoPE ``position_ids`` ``(batch, 3, seq_len)`` with
       ``[temporal, height, width]`` channels, matching HF's
       ``get_rope_index``.

    Args:
        tokenizer: A ``Qwen2VLTokenizer`` instance. Vision special token
            IDs (``image_token``, ``video_token``, etc.) are resolved from
            the tokenizer's vocabulary automatically.
        image_converter: A ``Qwen2VLImageConverter`` instance, or ``None``
            for text-only mode.
        sequence_length: int. Total padded sequence length. Default 1024.
        add_start_token: bool. Prepend BOS token. Default ``False``.
        add_end_token: bool. Append EOS token. Default ``True``.
    """

    backbone_cls = Qwen2VLBackbone
    tokenizer_cls = Qwen2VLTokenizer
    image_converter_cls = Qwen2VLImageConverter

    _SPECIAL_TOKEN_ATTRS = [
        "im_start_token",
        "end_token",
        "vision_start_token",
        "vision_end_token",
        "image_token",
        "video_token",
    ]

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

        # Token strings — these are static.
        self.image_token = getattr(
            self.tokenizer, "image_token", "<|image_pad|>"
        )
        self.video_token = getattr(
            self.tokenizer, "video_token", "<|video_pad|>"
        )

        # Lazily built after the tokenizer's vocabulary is loaded.
        self._cached_special_token_map = None
        self._cached_special_token_pattern = None

    @property
    def image_token_id(self):
        """Image pad token ID, resolved from the tokenizer."""
        return getattr(self.tokenizer, "image_token_id", None)

    @property
    def video_token_id(self):
        """Video pad token ID, resolved from the tokenizer."""
        return getattr(self.tokenizer, "video_token_id", None)

    @property
    def _special_token_map(self):
        """Lazily build token-string → token-ID map."""
        if self._cached_special_token_map is None:
            self._cached_special_token_map = {}
            for attr in self._SPECIAL_TOKEN_ATTRS:
                tok_str = getattr(self.tokenizer, attr, None)
                tok_id = getattr(self.tokenizer, f"{attr}_id", None)
                if tok_str is not None and tok_id is not None:
                    self._cached_special_token_map[tok_str] = tok_id
        return self._cached_special_token_map

    @property
    def _special_token_pattern(self):
        """Lazily build regex for splitting at special tokens."""
        if self._cached_special_token_pattern is None:
            self._cached_special_token_pattern = re.compile(
                "("
                + "|".join(re.escape(t) for t in self._special_token_map)
                + ")"
            )
        return self._cached_special_token_pattern

    def _tokenize_with_special_tokens(
        self, text, num_image_tokens, num_video_tokens
    ):
        """Tokenize text while correctly handling special tokens.

        The KerasHub BPE tokenizer may not encode Qwen2-VL's added special
        tokens (``<|image_pad|>``, ``<|vision_start|>``, etc.) as single
        tokens — it can break them into sub-word pieces. This method
        splits the input by known special tokens, tokenizes only the
        text segments, and manually inserts the correct token IDs.

        Each ``<|image_pad|>``/``<|video_pad|>`` marker is expanded to the
        number of merged vision tokens for the corresponding image/video.

        Args:
            text: str. The prompt string.
            num_image_tokens: list[int]. Merged token count per image.
            num_video_tokens: list[int]. Merged token count per video.
        Returns:
            list[int]. The complete token ID sequence.
        """
        parts = self._special_token_pattern.split(text)

        all_ids = []
        img_idx = 0
        vid_idx = 0
        for part in parts:
            if part in self._special_token_map:
                if part == self.image_token:
                    if img_idx < len(num_image_tokens):
                        n = num_image_tokens[img_idx]
                        img_idx += 1
                    else:
                        n = 1
                    all_ids.extend([self.image_token_id] * n)
                elif part == self.video_token:
                    if vid_idx < len(num_video_tokens):
                        n = num_video_tokens[vid_idx]
                        vid_idx += 1
                    else:
                        n = 1
                    all_ids.extend([self.video_token_id] * n)
                else:
                    all_ids.append(self._special_token_map[part])
            elif part:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    def _compute_vision_indices(self, token_ids):
        """Return indices where token_ids matches image or video token IDs.

        Indices are strictly ordered: all image token indices followed by all
        video token indices. This matches the concatenated order of
        ``pixel_values``.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
        Returns:
            int32 tensor ``(total_vision_tokens,)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        img_indices = np.where(
            (token_ids_np == self.image_token_id).reshape(-1)
        )[0]
        vid_indices = np.where(
            (token_ids_np == self.video_token_id).reshape(-1)
        )[0]
        return tf.constant(
            np.concatenate([img_indices, vid_indices]).astype("int32")
        )

    def _compute_position_ids(self, token_ids, image_grid_thw, video_grid_thw):
        """Build 3-channel M-RoPE position IDs matching HF's algorithm.

        For text tokens all 3 channels hold the same sequential position.
        For vision tokens the channels encode ``[temporal, height, width]``
        grid coordinates:

        - temporal: ``current_pos + frame_index``
        - height: ``current_pos + h_idx`` (``h_idx`` in merged grid)
        - width: ``current_pos + w_idx`` (``w_idx`` in merged grid)
        - ``current_pos`` advances by ``max(h, w) // merge_size`` after each
          vision span (HF ``get_rope_index`` convention).

        Positions continue sequentially into the padded region so that a
        decode step can simply slice ``position_ids[:, :, i]`` — equivalent
        to HF's ``mrope_position_deltas`` bookkeeping.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
            image_grid_thw: int32 tensor ``(num_images, 3)`` or ``None``.
            video_grid_thw: int32 tensor ``(num_videos, 3)`` or ``None``.
        Returns:
            int32 tensor ``(batch, 3, seq_len)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)

        def _grid_to_numpy(grid):
            if grid is None:
                return np.zeros((0, 3), dtype=np.int32)
            if hasattr(grid, "numpy"):
                return ops.convert_to_numpy(grid)
            return np.array(grid)

        image_grid_np = _grid_to_numpy(image_grid_thw)
        video_grid_np = _grid_to_numpy(video_grid_thw)

        batch_size, seq_len = token_ids_np.shape
        merge_size = getattr(self.image_converter, "merge_size", 2)

        all_pos = np.zeros((batch_size, 3, seq_len), dtype=np.int32)

        for b in range(batch_size):
            ids = token_ids_np[b]
            t_pos = np.zeros(seq_len, dtype=np.int32)
            h_pos = np.zeros(seq_len, dtype=np.int32)
            w_pos = np.zeros(seq_len, dtype=np.int32)

            current_pos = 0
            img_idx = 0
            vid_idx = 0
            i = 0
            while i < seq_len:
                is_image = (
                    ids[i] == self.image_token_id
                    and img_idx < image_grid_np.shape[0]
                )
                is_video = (
                    ids[i] == self.video_token_id
                    and vid_idx < video_grid_np.shape[0]
                )

                if is_image or is_video:
                    if is_image:
                        t_grid = int(image_grid_np[img_idx, 0])
                        h_grid = int(image_grid_np[img_idx, 1])
                        w_grid = int(image_grid_np[img_idx, 2])
                        img_idx += 1
                    else:
                        t_grid = int(video_grid_np[vid_idx, 0])
                        h_grid = int(video_grid_np[vid_idx, 1])
                        w_grid = int(video_grid_np[vid_idx, 2])
                        vid_idx += 1

                    llm_grid_h = h_grid // merge_size
                    llm_grid_w = w_grid // merge_size
                    frame_tokens = llm_grid_h * llm_grid_w
                    n_tokens = t_grid * frame_tokens

                    span_end = min(i + n_tokens, seq_len)
                    for vi in range(span_end - i):
                        frame = vi // frame_tokens
                        h_idx = (vi // llm_grid_w) % llm_grid_h
                        w_idx = vi % llm_grid_w
                        t_pos[i + vi] = current_pos + frame
                        h_pos[i + vi] = current_pos + h_idx
                        w_pos[i + vi] = current_pos + w_idx

                    current_pos += max(llm_grid_h, llm_grid_w)
                    i = span_end
                else:
                    t_pos[i] = current_pos
                    h_pos[i] = current_pos
                    w_pos[i] = current_pos
                    current_pos += 1
                    i += 1

            all_pos[b, 0] = t_pos
            all_pos[b, 1] = h_pos
            all_pos[b, 2] = w_pos

        return tf.constant(all_pos, dtype="int32")

    def _preprocess_images(self, images):
        """Convert raw images to patch tensors using the image converter.

        Args:
            images: A single image ``(H, W, C)``, a list of images, or a
                batched array ``(B, H, W, C)``.
        Returns:
            dict with ``pixel_values`` ``(total_patches, T, pH, pW, C)`` and
            ``image_grid_thw`` ``(num_images, 3)``.
        """
        # Normalize to a flat list of individual 3-D images. A list of
        # differently-shaped images arrives as a `tf.RaggedTensor` after
        # preprocessing input conversion.
        if isinstance(images, tf.RaggedTensor):
            flat_images = [np.array(img.to_list()) for img in images]
        elif isinstance(images, (list, tuple)):
            flat_images = []
            for img in images:
                if hasattr(img, "shape") and len(img.shape) == 4:
                    for i in range(img.shape[0]):
                        flat_images.append(img[i])
                else:
                    flat_images.append(img)
        elif hasattr(images, "shape") and len(images.shape) == 4:
            flat_images = [images[i] for i in range(images.shape[0])]
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
                patches = tf.constant(patches)
            if not isinstance(grid_thw, tf.Tensor):
                grid_thw = tf.constant(grid_thw)

            all_patches.append(patches)
            all_grid_thw.append(grid_thw)

        return {
            "pixel_values": tf.concat(all_patches, axis=0),
            "grid_thw": tf.concat(all_grid_thw, axis=0),
        }

    def _preprocess_videos(self, videos):
        """Convert raw videos to patch tensors using the image converter.

        Args:
            videos: A single video ``(T, H, W, C)``, a list of videos, or a
                batched array ``(B, T, H, W, C)``.
        Returns:
            dict with ``pixel_values`` ``(total_patches, T, pH, pW, C)`` and
            ``grid_thw`` ``(num_videos, 3)``.
        """
        if isinstance(videos, tf.RaggedTensor):
            flat_videos = [np.array(vid.to_list()) for vid in videos]
        elif isinstance(videos, (list, tuple)):
            flat_videos = []
            for vid in videos:
                if hasattr(vid, "shape") and len(vid.shape) == 5:
                    for i in range(vid.shape[0]):
                        flat_videos.append(vid[i])
                else:
                    flat_videos.append(vid)
        elif hasattr(videos, "shape") and len(videos.shape) == 5:
            flat_videos = [videos[i] for i in range(videos.shape[0])]
        else:
            flat_videos = [videos]

        all_patches = []
        all_grid_thw = []
        for vid in flat_videos:
            result = self.image_converter(vid)
            patches = result["patches"]
            grid_thw = result["grid_thw"]

            if not isinstance(patches, tf.Tensor):
                patches = tf.constant(patches)
            if not isinstance(grid_thw, tf.Tensor):
                grid_thw = tf.constant(grid_thw)

            all_patches.append(patches)
            all_grid_thw.append(grid_thw)

        return {
            "pixel_values": tf.concat(all_patches, axis=0),
            "grid_thw": tf.concat(all_grid_thw, axis=0),
        }

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """Preprocess inputs for generation (prompt-only, no labels).

        Accepts either:
        - A plain string / list of strings (text-only).
        - A dict with ``"prompts"`` and optional ``"images"`` and
          ``"videos"`` keys.

        Returns:
            dict with ``token_ids``, ``padding_mask``, and, for multimodal
            inputs, ``pixel_values``, ``image_grid_thw``,
            ``vision_indices``, ``position_ids``.
        """
        # Check whether the input has images/videos.
        images = None
        videos = None
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)

        # Text-only: delegate to the base class entirely.
        if images is None and videos is None:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        # Multimodal path.
        assert_tf_installed("Qwen2VLCausalLMPreprocessor with images/videos")
        if not self.built:
            self.build(None)
        if self.image_converter is None:
            raise ValueError(
                "This preprocessor was created without an "
                "`image_converter`, so it cannot process images or videos. "
                "To process multimodal inputs, pass an "
                "`image_converter` when creating the preprocessor, or "
                "load a preset that includes one."
            )

        sequence_length = sequence_length or self.sequence_length
        prompts = x["prompts"]

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, 0)

        # 1. Process images and videos.
        vision_out_images = None
        vision_out_videos = None
        if images is not None:
            vision_out_images = self._preprocess_images(images)
        if videos is not None:
            vision_out_videos = self._preprocess_videos(videos)

        # 2. Compute merged token counts per image/video.
        merge_size = getattr(self.image_converter, "merge_size", 2)

        num_image_tokens = []
        if vision_out_images is not None:
            grid_np = ops.convert_to_numpy(vision_out_images["grid_thw"])
            for i in range(grid_np.shape[0]):
                t = int(grid_np[i, 0])
                h = int(grid_np[i, 1])
                w = int(grid_np[i, 2])
                num_image_tokens.append(
                    t * (h // merge_size) * (w // merge_size)
                )

        num_video_tokens = []
        if vision_out_videos is not None:
            grid_np = ops.convert_to_numpy(vision_out_videos["grid_thw"])
            for i in range(grid_np.shape[0]):
                t = int(grid_np[i, 0])
                h = int(grid_np[i, 1])
                w = int(grid_np[i, 2])
                num_video_tokens.append(
                    t * (h // merge_size) * (w // merge_size)
                )

        # 3. Normalize prompts to a list of python strings.
        if isinstance(prompts, tf.Tensor):
            prompts_list = [p.numpy().decode("utf-8") for p in prompts]
        elif isinstance(prompts, (list, tuple)):
            prompts_list = [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ]
        else:
            prompts_list = [str(prompts)]

        # 4. Tokenize with special-token-aware splitting.
        expanded_sequences = [
            self._tokenize_with_special_tokens(
                prompt_str, num_image_tokens, num_video_tokens
            )
            for prompt_str in prompts_list
        ]

        # 5. Pack to fixed length.
        token_ids_ragged = tf.ragged.constant(expanded_sequences, dtype="int32")
        token_ids, padding_mask = self.packer(
            token_ids_ragged,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        # 6. Compute vision indices & M-RoPE position IDs.
        vision_indices = self._compute_vision_indices(token_ids)

        img_grid = vision_out_images["grid_thw"] if vision_out_images else None
        vid_grid = vision_out_videos["grid_thw"] if vision_out_videos else None
        pos_ids = self._compute_position_ids(token_ids, img_grid, vid_grid)

        # 7. Build combined pixel_values / image_grid_thw for the encoder.
        pixel_values_list = []
        grid_list = []
        if vision_out_images is not None:
            pixel_values_list.append(vision_out_images["pixel_values"])
            grid_list.append(vision_out_images["grid_thw"])
        if vision_out_videos is not None:
            pixel_values_list.append(vision_out_videos["pixel_values"])
            grid_list.append(vision_out_videos["grid_thw"])

        combined_pixel_values = tf.concat(pixel_values_list, axis=0)
        combined_grid_thw = tf.concat(grid_list, axis=0)

        return {
            "token_ids": token_ids if batched else tf.squeeze(token_ids, 0),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, 0)
            ),
            "pixel_values": combined_pixel_values,
            "image_grid_thw": combined_grid_thw,
            "vision_indices": vision_indices,
            "position_ids": (pos_ids if batched else tf.squeeze(pos_ids, 0)),
        }

    def _generate_postprocess_python(self, x):
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids = keras.ops.convert_to_numpy(token_ids).astype("int32")
        padding_mask = keras.ops.convert_to_numpy(padding_mask).astype("bool")

        # Collect all IDs to strip: base special tokens + vision tokens.
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self._special_token_map.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)

        if token_ids.ndim == 1:
            mask = padding_mask
            for tok_id in ids_to_strip:
                mask = mask & (token_ids != tok_id)
            token_ids = token_ids[mask].tolist()
        else:
            ragged_ids = []
            for i in range(token_ids.shape[0]):
                mask = padding_mask[i]
                for tok_id in ids_to_strip:
                    mask = mask & (token_ids[i] != tok_id)
                ragged_ids.append(token_ids[i][mask].tolist())
            token_ids = ragged_ids
        return self.tokenizer.detokenize(token_ids)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        if not self.built:
            self.build(None)

        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        padding_mask = keras.ops.convert_to_numpy(x["padding_mask"])

        # Collect all IDs to strip: base special tokens + vision tokens.
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self._special_token_map.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        output = self.tokenizer.detokenize(token_ids)

        # Safety net: strip residual special token strings that may
        # survive if the BPE model encodes them as byte-fallback pieces.
        for tok_str in self._special_token_map:
            output = tf.strings.regex_replace(output, re.escape(tok_str), "")
        return output

    def get_config(self):
        config = super().get_config()
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        return config
