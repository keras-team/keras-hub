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
from keras_hub.src.models.qwen3_5.qwen3_5_video_converter import (
    Qwen3_5VideoConverter,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


@keras_hub_export("keras_hub.models.Qwen3_5CausalLMPreprocessor")
class Qwen3_5CausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3.5 Causal LM preprocessor with multimodal support.

    For text-only usage this behaves identically to the base
    ``CausalLMPreprocessor``.  When an ``image_converter`` is provided,
    the preprocessor also:

    1. Converts images to patch tensors via ``Qwen3_5ImageConverter``.
    2. Replaces ``<|image_pad|>`` and ``<|video_pad|>`` placeholder tokens
       in the token sequence with the correct number of vision tokens.
    3. Computes flat ``vision_indices`` for scattering visual embeddings
       into the text sequence.
    4. Builds 4-channel M-RoPE ``position_ids`` for spatial awareness.

    Args:
        tokenizer: A ``Qwen3_5Tokenizer`` instance.
        image_converter: A ``Qwen3_5ImageConverter`` instance, or ``None``
            for text-only mode.
        video_converter: A ``Qwen3_5VideoConverter`` instance, or ``None``.
        sequence_length: int. Total padded sequence length. Default 1024.
        add_start_token: bool. Prepend BOS token. Default ``False``.
        add_end_token: bool. Append EOS token. Default ``True``.
        image_token: str. The placeholder token the user inserts in prompts
            to indicate where an image should go. Default ``"<|image_pad|>"``.
        image_token_id: int. The token ID of ``image_token``. Default 248056.
        video_token: str. The placeholder token for videos. Default
            ``"<|video_pad|>"``.
        video_token_id: int. The token ID of ``video_token``. Default 248057.
    """

    backbone_cls = Qwen3_5Backbone
    tokenizer_cls = Qwen3_5Tokenizer
    image_converter_cls = Qwen3_5ImageConverter
    video_converter_cls = Qwen3_5VideoConverter

    # Special tokens that the KerasHub BPE tokenizer may not encode
    # as single tokens. Map from string → token ID.
    SPECIAL_TOKEN_MAP = {
        "<|im_start|>": 248045,
        "<|im_end|>": 248044,
        "<|vision_start|>": 248053,
        "<|vision_end|>": 248054,
        "<|image_pad|>": 248056,
        "<|video_pad|>": 248057,
    }

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        image_token="<|image_pad|>",
        image_token_id=248056,
        video_token="<|video_pad|>",
        video_token_id=248057,
        video_fps=2.0,
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
        self.video_converter = video_converter
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.video_token = video_token
        self.video_token_id = video_token_id
        self.video_fps = video_fps

        # Regex pattern for splitting prompts at special token boundaries.
        self._special_token_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in self.SPECIAL_TOKEN_MAP) + ")"
        )

    def _tokenize_with_special_tokens(
        self, text, num_image_tokens, num_video_tokens
    ):
        """Tokenize text while correctly handling special tokens.

        The KerasHub BPE tokenizer may not encode Qwen3.5's added special
        tokens (``<|image_pad|>``, ``<|vision_start|>``, etc.) as single
        tokens — it can break them into sub-word pieces. This method
        splits the input by known special tokens, tokenizes only the
        text segments, and manually inserts the correct token IDs.

        For ``<|image_pad|>`` and ``<|video_pad|>`` tokens, each occurrence is
        expanded to ``N`` copies.

        Args:
            text: str. The prompt string.
            num_image_tokens: list[int].
            num_video_tokens: list[int].
        Returns:
            list[int]. The complete token ID sequence.
        """
        parts = self._special_token_pattern.split(text)

        all_ids = []
        img_idx = 0
        vid_idx = 0
        for part in parts:
            if part in self.SPECIAL_TOKEN_MAP:
                if part == self.image_token:
                    # Expand image placeholder to N copies.
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
                    all_ids.append(self.SPECIAL_TOKEN_MAP[part])
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
        video indices. This matches the concatenated order of `pixel_values`.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
        Returns:
            int32 tensor ``(total_vision_tokens,)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        img_mask = (token_ids_np == self.image_token_id).reshape(-1)
        img_indices = np.where(img_mask)[0].astype(np.int32)

        vid_mask = (token_ids_np == self.video_token_id).reshape(-1)
        vid_indices = np.where(vid_mask)[0].astype(np.int32)

        return tf.constant(np.concatenate([img_indices, vid_indices], axis=0))

    def _expand_video_prompt(
        self, prompt, video_grid_thws, temporal_patch_size=2
    ):
        """Expand ``<|vision_start|><|video_pad|><|vision_end|>`` into
        per-frame sections with timestamps.

        Matches HF's Qwen3.5 processor which produces::

            <0.5 seconds><|vision_start|><|video_pad|>×N<|vision_end|>
            <1.0 seconds><|vision_start|><|video_pad|>×N<|vision_end|>

        Each frame gets its own ``<|vision_start|>...<|vision_end|>``
        block with a timestamp prefix.

        Args:
            prompt: str. The raw prompt string.
            video_grid_thws: list of (T, H, W) tuples, one per video.
            temporal_patch_size: int. Frames per temporal patch.
        Returns:
            tuple of (expanded_prompt, num_video_tokens_per_frame).
        """
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )
        video_marker = "<|vision_start|><|video_pad|><|vision_end|>"
        num_video_tokens_per_frame = []

        for grid_thw in video_grid_thws:
            t_grid, h_grid, w_grid = (
                int(grid_thw[0]),
                int(grid_thw[1]),
                int(grid_thw[2]),
            )
            frame_seqlen = (h_grid // merge_size) * (w_grid // merge_size)

            # Compute per-frame timestamps (average of grouped frames).
            # HF: timestamps[i] = (idx_start + idx_end) / 2 / fps
            # We approximate with evenly-spaced timestamps.
            timestamps = []
            for frame_idx in range(t_grid):
                # Each temporal patch groups `temporal_patch_size` raw frames.
                start_raw = frame_idx * temporal_patch_size
                end_raw = start_raw + temporal_patch_size - 1
                ts = (start_raw + end_raw) / 2.0 / self.video_fps
                timestamps.append(ts)

            # Build the per-frame block.
            video_block = ""
            for frame_idx in range(t_grid):
                video_block += f"<{timestamps[frame_idx]:.1f} seconds>"
                video_block += "<|vision_start|><|video_pad|><|vision_end|>"
                num_video_tokens_per_frame.append(frame_seqlen)

            prompt = prompt.replace(video_marker, video_block, 1)

        return prompt, num_video_tokens_per_frame

    def _compute_position_ids(self, token_ids, image_grid_thw, video_grid_thw):
        """Build 4-channel M-RoPE position IDs matching HF's algorithm.

        For text tokens all 4 channels have the same sequential position.
        For vision tokens channels 1-3 encode (temporal, height, width)
        grid coordinates. Channel 0 mirrors channel 1 (temporal).

        This matches HF's ``get_rope_index`` / ``get_vision_position_ids``:
        - temporal: ``full(n_tokens, start_pos)``
        - height: ``arange(h_merged).repeat_interleave(w_merged * t)``
        - width: ``arange(w_merged).repeat(h_merged * t)``
        - text_pos advances by ``max(h_merged, w_merged)`` after vision span.

        For **video**, grids are **split per-frame** before processing
        (matching HF's ``repeat_interleave + set T=1`` pattern), so each
        per-frame ``<|video_pad|>`` group is treated as a separate vision
        span with ``T=1``.

        Args:
            token_ids: int32 tensor ``(batch, seq_len)``.
            image_grid_thw: int32 tensor ``(num_images, 3)``.
            video_grid_thw: int32 tensor ``(num_videos, 3)``.
        Returns:
            int32 tensor ``(batch, 4, seq_len)``.
        """
        token_ids_np = ops.convert_to_numpy(token_ids)
        if hasattr(image_grid_thw, "numpy"):
            image_grid_np = ops.convert_to_numpy(image_grid_thw)
        elif image_grid_thw is not None:
            image_grid_np = np.array(image_grid_thw)
        else:
            image_grid_np = np.zeros((0, 3), dtype=np.int32)

        if hasattr(video_grid_thw, "numpy"):
            video_grid_np = ops.convert_to_numpy(video_grid_thw)
        elif video_grid_thw is not None:
            video_grid_np = np.array(video_grid_thw)
        else:
            video_grid_np = np.zeros((0, 3), dtype=np.int32)

        # Split video grids per-frame: [T, H, W] → T rows of [1, H, W].
        # This matches HF's get_rope_index which does:
        #   video_grid_thw = repeat_interleave(video_grid_thw, T, dim=0)
        #   video_grid_thw[:, 0] = 1
        if video_grid_np.shape[0] > 0:
            expanded = []
            for i in range(video_grid_np.shape[0]):
                t = int(video_grid_np[i, 0])
                for _ in range(t):
                    expanded.append(
                        [1, video_grid_np[i, 1], video_grid_np[i, 2]]
                    )
            video_grid_np = np.array(expanded, dtype=np.int32)

        batch_size, seq_len = token_ids_np.shape
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

        all_pos = np.zeros((batch_size, 4, seq_len), dtype=np.int32)

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
                if (
                    ids[i] == self.image_token_id
                    and img_idx < image_grid_np.shape[0]
                ):
                    t_grid = int(image_grid_np[img_idx, 0])
                    h_grid = int(image_grid_np[img_idx, 1])
                    w_grid = int(image_grid_np[img_idx, 2])
                    img_idx += 1
                    is_vision = True
                elif (
                    ids[i] == self.video_token_id
                    and vid_idx < video_grid_np.shape[0]
                ):
                    t_grid = int(video_grid_np[vid_idx, 0])
                    h_grid = int(video_grid_np[vid_idx, 1])
                    w_grid = int(video_grid_np[vid_idx, 2])
                    vid_idx += 1
                    is_vision = True
                else:
                    is_vision = False

                if is_vision:
                    llm_grid_t = t_grid
                    llm_grid_h = h_grid // merge_size
                    llm_grid_w = w_grid // merge_size
                    n_tokens = llm_grid_t * llm_grid_h * llm_grid_w
                    span_end = min(i + n_tokens, seq_len)

                    # Replicate HF's get_vision_position_ids exactly:
                    for vi in range(span_end - i):
                        t_pos[i + vi] = current_pos
                        row = vi // llm_grid_w
                        col = vi % llm_grid_w
                        h_pos[i + vi] = current_pos + (row % llm_grid_h)
                        w_pos[i + vi] = current_pos + col

                    # Advance by max(h_merged, w_merged) — HF convention.
                    current_pos += max(llm_grid_h, llm_grid_w)
                    i = span_end
                else:
                    t_pos[i] = current_pos
                    h_pos[i] = current_pos
                    w_pos[i] = current_pos
                    current_pos += 1
                    i += 1

            # Channel layout: [text, temporal, height, width].
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

        # 1. Process images and videos
        vision_out_images = None
        vision_out_videos = None

        if images is not None and self.image_converter is not None:
            vision_out_images = self._preprocess_images(images, batched)
        if videos is not None and self.video_converter is not None:
            vision_out_videos = self._preprocess_videos(videos, batched)

        # 2. Compute token counts for images.
        merge_size = getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

        num_image_tokens = []
        if vision_out_images is not None:
            grid_np = (
                vision_out_images["image_grid_thw"].numpy()
                if hasattr(vision_out_images["image_grid_thw"], "numpy")
                else np.array(vision_out_images["image_grid_thw"])
            )
            for i in range(grid_np.shape[0]):
                t = int(grid_np[i, 0])
                h = int(grid_np[i, 1])
                w = int(grid_np[i, 2])
                num_image_tokens.append(
                    t * (h // merge_size) * (w // merge_size)
                )

        # 3. Build prompt strings and expand video tokens per-frame.
        if isinstance(prompts, tf.Tensor):
            prompts_list = [p.numpy().decode("utf-8") for p in prompts]
        elif isinstance(prompts, (list, tuple)):
            prompts_list = [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ]
        else:
            prompts_list = [str(prompts)]

        # Expand video prompts: replace <vision_start><video_pad><vision_end>
        # with per-frame timestamp sections.
        num_video_tokens = []
        if vision_out_videos is not None:
            grid_np = (
                vision_out_videos["grid_thw"].numpy()
                if hasattr(vision_out_videos["grid_thw"], "numpy")
                else np.array(vision_out_videos["grid_thw"])
            )
            video_grid_thws = [
                (int(grid_np[i, 0]), int(grid_np[i, 1]), int(grid_np[i, 2]))
                for i in range(grid_np.shape[0])
            ]
            temporal_patch_size = getattr(
                self.video_converter, "temporal_patch_size", 2
            )
            for idx in range(len(prompts_list)):
                prompts_list[idx], per_frame_tokens = self._expand_video_prompt(
                    prompts_list[idx],
                    video_grid_thws,
                    temporal_patch_size,
                )
                num_video_tokens.extend(per_frame_tokens)

        # 4. Tokenize with special-token-aware splitting.
        expanded_sequences = []
        for prompt_str in prompts_list:
            ids = self._tokenize_with_special_tokens(
                prompt_str, num_image_tokens, num_video_tokens
            )
            expanded_sequences.append(ids)

        # 5. Pack to fixed length.
        token_ids_ragged = tf.ragged.constant(expanded_sequences, dtype="int32")
        token_ids, padding_mask = self.packer(
            token_ids_ragged,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        # 6. Compute vision indices & M-RoPE position IDs.
        vision_indices = self._compute_vision_indices(token_ids)

        img_grid = (
            vision_out_images["image_grid_thw"] if vision_out_images else None
        )
        vid_grid = vision_out_videos["grid_thw"] if vision_out_videos else None
        pos_ids = self._compute_position_ids(token_ids, img_grid, vid_grid)

        # 7. Build combined pixel_values / image_grid_thw for vision encoder.
        pixel_values_list = []
        grid_list = []
        if vision_out_images is not None:
            pixel_values_list.append(vision_out_images["pixel_values"])
            grid_list.append(vision_out_images["image_grid_thw"])
        if vision_out_videos is not None:
            pixel_values_list.append(vision_out_videos["patches"])
            grid_list.append(vision_out_videos["grid_thw"])

        combined_pixel_values = (
            tf.concat(pixel_values_list, axis=0) if pixel_values_list else None
        )
        combined_grid_thw = tf.concat(grid_list, axis=0) if grid_list else None

        result = {
            "token_ids": token_ids if batched else tf.squeeze(token_ids, 0),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, 0)
            ),
        }
        if combined_pixel_values is not None:
            result["pixel_values"] = combined_pixel_values
            result["image_grid_thw"] = combined_grid_thw

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

    def _preprocess_videos(self, videos, batched):
        """Convert raw videos to patch tensors using the video converter."""
        if isinstance(videos, (list, tuple)):
            flat_videos = []
            for vid in videos:
                if hasattr(vid, "shape") and len(vid.shape) == 5:
                    for i in range(vid.shape[0]):
                        flat_videos.append(vid[i])
                else:
                    flat_videos.append(vid)
        elif hasattr(videos, "shape") and len(videos.shape) == 5:
            flat_videos = [videos[i] for i in range(videos.shape[0])]
        elif hasattr(videos, "shape") and len(videos.shape) == 4:
            flat_videos = [videos]
        else:
            flat_videos = [videos]

        all_patches = []
        all_grid_thw = []
        for vid in flat_videos:
            result = self.video_converter(vid)
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
            "patches": tf.concat(all_patches, axis=0),
            "grid_thw": tf.stack(all_grid_thw, axis=0),
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
                "video_token": self.video_token,
                "video_token_id": self.video_token_id,
                "video_fps": self.video_fps,
            }
        )
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        if self.video_converter is not None:
            config["video_converter"] = keras.layers.serialize(
                self.video_converter
            )
        return config
