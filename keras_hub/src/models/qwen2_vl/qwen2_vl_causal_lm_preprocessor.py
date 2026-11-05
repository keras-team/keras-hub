import re

import keras
import numpy as np
from keras import ops

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
from keras_hub.src.models.qwen2_vl.qwen2_vl_video_converter import (
    Qwen2VLVideoConverter,
)
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    QwenVLSpecialTokenMixin,
)
from keras_hub.src.utils.tensor_utils import assert_tf_installed
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Qwen2VLCausalLMPreprocessor")
class Qwen2VLCausalLMPreprocessor(
    QwenVLSpecialTokenMixin, CausalLMPreprocessor
):
    """Qwen2-VL Causal LM preprocessor with multimodal support.

    For text-only usage this behaves identically to the base
    ``CausalLMPreprocessor``. When an ``image_converter`` is provided, the
    preprocessor also:

    1. Converts images/videos to patch tensors via
       ``Qwen2VLImageConverter``/``Qwen2VLVideoConverter``.
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
        video_converter: A ``Qwen2VLVideoConverter`` instance, or ``None``.
            Falls back to ``image_converter`` when unset, since Qwen2-VL
            applies identical preprocessing to video frames and images.
        sequence_length: int. Total padded sequence length. Default 1024.
        add_start_token: bool. Prepend BOS token. Default ``False``.
        add_end_token: bool. Append EOS token. Default ``True``.
    """

    backbone_cls = Qwen2VLBackbone
    tokenizer_cls = Qwen2VLTokenizer
    image_converter_cls = Qwen2VLImageConverter
    video_converter_cls = Qwen2VLVideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
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
        self.video_converter = video_converter

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

    def build(self, input_shape):
        super().build(input_shape)
        self.multi_packer = MultiSegmentPacker(
            start_value=[],
            end_value=self.tokenizer.end_token_id or [],
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )

    def _check_converters(self, images, videos):
        """Raise when media inputs arrive without a converter to process."""
        if images is not None and self.image_converter is None:
            raise ValueError(
                "This preprocessor was created without an "
                "`image_converter`, so it cannot process images. "
                "To process image inputs, pass an `image_converter` when "
                "creating the preprocessor, or load a preset that includes "
                "one."
            )
        if (
            videos is not None
            and self.image_converter is None
            and (self.video_converter is None)
        ):
            raise ValueError(
                "This preprocessor was created without an "
                "`image_converter` or `video_converter`, so it cannot "
                "process videos. To process video inputs, pass a converter "
                "when creating the preprocessor, or load a preset that "
                "includes one."
            )

    def _flatten_prompts(self, prompts):
        """Normalize a string / Tensor / list of prompts to a list of str."""
        if isinstance(prompts, str):
            return [prompts], False
        if isinstance(prompts, tf.Tensor):
            if len(prompts.shape) == 0:
                return [prompts.numpy().decode("utf-8")], False
            return [p.numpy().decode("utf-8") for p in prompts], True
        return [
            p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
            for p in prompts
        ], True

    def _process_media(self, images, videos):
        """Run converters and count merged tokens per media item."""
        vision_out_images = (
            self._preprocess_images(images) if images is not None else None
        )
        vision_out_videos = (
            self._preprocess_videos(videos) if videos is not None else None
        )
        vision_converter = self.image_converter or self.video_converter
        merge_size = getattr(vision_converter, "merge_size", 2)

        def _counts(vision_out):
            if vision_out is None:
                return []
            grid_np = ops.convert_to_numpy(vision_out["grid_thw"])
            return [
                int(row[0])
                * (int(row[1]) // merge_size)
                * (int(row[2]) // merge_size)
                for row in grid_np
            ]

        return (
            vision_out_images,
            vision_out_videos,
            _counts(vision_out_images),
            _counts(vision_out_videos),
        )

    def _check_sequence_length(self, expanded_sequences, sequence_length):
        """Raise if an expanded sequence cannot fit `sequence_length`.

        Truncating mid vision span would desynchronize `vision_indices`
        from `pixel_values`, so fail loudly instead of truncating.
        """
        longest = max(len(s) for s in expanded_sequences)
        if longest > sequence_length:
            raise ValueError(
                f"Expanded input requires at least {longest} tokens, but "
                f"`sequence_length` is {sequence_length}. Increase "
                "`sequence_length` to fit the vision token spans."
            )

    def _expand_prompts(self, prompts_list, num_image_tokens, num_video_tokens):
        """Tokenize prompts, expanding media markers in prompt order."""
        n_img = sum(p.count(self.image_token) for p in prompts_list)
        n_vid = sum(p.count(self.video_token) for p in prompts_list)
        if n_img != len(num_image_tokens):
            raise ValueError(
                f"Found {n_img} `{self.image_token}` placeholder(s) in the "
                f"prompts but {len(num_image_tokens)} image(s) were "
                "provided."
            )
        if n_vid != len(num_video_tokens):
            raise ValueError(
                f"Found {n_vid} `{self.video_token}` placeholder(s) in the "
                f"prompts but {len(num_video_tokens)} video(s) were "
                "provided."
            )
        expanded_sequences = []
        img_offset = 0
        vid_offset = 0
        for prompt_str in prompts_list:
            n_img = prompt_str.count(self.image_token)
            n_vid = prompt_str.count(self.video_token)
            expanded_sequences.append(
                self._tokenize_with_special_tokens(
                    prompt_str,
                    num_image_tokens[img_offset : img_offset + n_img],
                    num_video_tokens[vid_offset : vid_offset + n_vid],
                )
            )
            img_offset += n_img
            vid_offset += n_vid
        return expanded_sequences

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

        # Image/video grids are consumed in prompt order across the batch —
        # keep running offsets so each row indexes its own media's grids.
        img_idx = 0
        vid_idx = 0
        for b in range(batch_size):
            ids = token_ids_np[b]
            t_pos = np.zeros(seq_len, dtype=np.int32)
            h_pos = np.zeros(seq_len, dtype=np.int32)
            w_pos = np.zeros(seq_len, dtype=np.int32)

            current_pos = 0
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
        """Convert raw videos to patch tensors using the video converter.

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

        converter = self.video_converter or self.image_converter
        all_patches = []
        all_grid_thw = []
        for vid in flat_videos:
            result = converter(vid)
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
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Text-only fast path for non-dict inputs.
        if not isinstance(x, dict):
            return super().call(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )

        assert_tf_installed("Qwen2VLCausalLMPreprocessor")
        if not self.built:
            self.build(None)

        images = x.get("images", None)
        videos = x.get("videos", None)

        def _empty(media):
            # Empty collections behave like absent media, so placeholder
            # counts get the descriptive mismatch error downstream.
            if isinstance(media, (list, tuple)):
                return len(media) == 0
            shape = getattr(media, "shape", None)
            return shape is not None and len(shape) > 0 and shape[0] == 0

        if _empty(images):
            images = None
        if _empty(videos):
            videos = None
        self._check_converters(images, videos)
        prompts_list, batched = self._flatten_prompts(x["prompts"])
        responses_text = x.get("responses", None)

        # Process media, expand markers, then pack prompts (+ responses).
        (
            vision_out_images,
            vision_out_videos,
            num_image_tokens,
            num_video_tokens,
        ) = self._process_media(images, videos)
        expanded_sequences = self._expand_prompts(
            prompts_list, num_image_tokens, num_video_tokens
        )
        self._check_sequence_length(expanded_sequences, sequence_length)

        if responses_text is not None:
            responses, _ = self._flatten_prompts(responses_text)
            response_ids = [
                self._tokenize_with_special_tokens(r, [], []) for r in responses
            ]
            prompt_ragged = tf.ragged.constant(
                expanded_sequences, dtype="int32"
            )
            response_ragged = tf.ragged.constant(response_ids, dtype="int32")
            token_ids, segment_ids = self.multi_packer(
                (prompt_ragged, response_ragged),
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            padding_mask = token_ids != self.tokenizer.pad_token_id
            response_mask = segment_ids == 1
            x_out = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            y = token_ids[..., 1:]
            sample_weight = response_mask[..., 1:]
        else:
            token_ids_ragged = tf.ragged.constant(
                expanded_sequences, dtype="int32"
            )
            token_ids, padding_mask = self.packer(
                token_ids_ragged,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            x_out = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]

        # 5. Vision indices, M-RoPE positions, and merged encoder inputs.
        if vision_out_images is not None or vision_out_videos is not None:
            vision_indices = self._compute_vision_indices(x_out["token_ids"])
            img_grid = (
                vision_out_images["grid_thw"] if vision_out_images else None
            )
            vid_grid = (
                vision_out_videos["grid_thw"] if vision_out_videos else None
            )
            pos_ids = self._compute_position_ids(
                x_out["token_ids"], img_grid, vid_grid
            )
            x_out["vision_indices"] = vision_indices
            x_out["position_ids"] = pos_ids

            pixel_values_list = []
            grid_list = []
            if vision_out_images is not None:
                pixel_values_list.append(vision_out_images["pixel_values"])
                grid_list.append(vision_out_images["grid_thw"])
            if vision_out_videos is not None:
                pixel_values_list.append(vision_out_videos["pixel_values"])
                grid_list.append(vision_out_videos["grid_thw"])
            x_out["pixel_values"] = tf.concat(pixel_values_list, axis=0)
            x_out["image_grid_thw"] = tf.concat(grid_list, axis=0)

        # 6. Squeeze batch dim for unbatched inputs.
        if not batched:
            for key in ("token_ids", "padding_mask", "position_ids"):
                if key in x_out:
                    x_out[key] = tf.squeeze(x_out[key], axis=0)
            y = tf.squeeze(y, axis=0)
            sample_weight = tf.squeeze(sample_weight, axis=0)

        return keras.utils.pack_x_y_sample_weight(x_out, y, sample_weight)

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

        # Text-only: delegate to the base class entirely. The base class
        # expects bare strings, so unwrap a `{"prompts": ...}` dict first.
        if images is None and videos is None:
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super().generate_preprocess(
                prompts, sequence_length=sequence_length
            )

        # Multimodal path.
        assert_tf_installed("Qwen2VLCausalLMPreprocessor with images/videos")
        if not self.built:
            self.build(None)
        self._check_converters(images, videos)

        sequence_length = sequence_length or self.sequence_length
        prompts_list, batched = self._flatten_prompts(x["prompts"])
        (
            vision_out_images,
            vision_out_videos,
            num_image_tokens,
            num_video_tokens,
        ) = self._process_media(images, videos)

        # Media are consumed in prompt order across the batch, so each
        # prompt gets the slice of token counts for the markers it contains.
        expanded_sequences = self._expand_prompts(
            prompts_list, num_image_tokens, num_video_tokens
        )

        # 5. Pack to fixed length.
        self._check_sequence_length(expanded_sequences, sequence_length)
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
        # `image_converter` and `video_converter` are serialized by the base
        # `Preprocessor.get_config`.
        return super().get_config()
