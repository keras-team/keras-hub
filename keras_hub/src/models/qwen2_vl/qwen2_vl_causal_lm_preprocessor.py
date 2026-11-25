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
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _flatten_media(node, leaf_ndim):
    """Recursively flatten arbitrarily nested media to leaf arrays.

    Media values may arrive as plain arrays, lists of arrays (with
    matching or differing shapes), batched tensors, or nested lists
    (e.g. media grouped per prompt through `generate()`). A leaf is an
    array of ``leaf_ndim`` or fewer dims — ``(H, W, C)``/``(H, W)`` for
    images, ``(T, H, W, C)`` for videos. Deeper arrays are split along
    axis 0; ragged nodes are flattened element-wise.
    """
    if not isinstance(node, (list, tuple)):
        try:
            # `convert_to_numpy` also handles backend tensors (e.g. CUDA)
            # that `np.asarray` cannot inspect.
            arr = convert_to_numpy(node)
        except (ValueError, TypeError, RuntimeError):
            arr = None
        if arr is None or arr.dtype == object:
            return [node]
        if arr.ndim <= leaf_ndim:
            return [arr]
        return [
            leaf
            for i in range(arr.shape[0])
            for leaf in _flatten_media(arr[i], leaf_ndim)
        ]
    try:
        arr = convert_to_numpy(node)
    except (ValueError, TypeError, RuntimeError):
        arr = None
    if arr is not None and arr.dtype != object:
        if arr.ndim > leaf_ndim:
            return _flatten_media(arr, leaf_ndim)
        if arr.ndim <= leaf_ndim and any(
            isinstance(elem, (list, tuple)) for elem in node
        ):
            # A single leaf expressed as nested lists (e.g. a
            # `RaggedTensor.to_list()` row), not a list of elements.
            return [arr]
    return [leaf for elem in node for leaf in _flatten_media(elem, leaf_ndim)]


def _empty_media(media):
    # Empty collections behave like absent media, so placeholder
    # counts get the descriptive mismatch error downstream.
    if isinstance(media, (list, tuple)):
        return len(media) == 0
    shape = getattr(media, "shape", None)
    return shape is not None and len(shape) > 0 and shape[0] == 0


def _has_media_markers(prompts_list, image_token, video_token):
    """True when any flattened string prompt carries a media marker."""
    return any(image_token in p or video_token in p for p in prompts_list)


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
            When unset, video preprocessing falls back to
            ``image_converter`` (shared, so bounds mutations propagate).
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
        # `None` defers to `self.image_converter` at use sites, so the
        # two media paths share one converter object and serialization
        # round-trips keep that identity.
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
        if (images is not None or videos is not None) and in_tf_function():
            raise NotImplementedError(
                "Image and video preprocessing is not supported inside "
                "`tf.function` or `tf.data` tracing yet (tracked in "
                "https://github.com/keras-team/keras-hub/issues/3226). "
                "Run eagerly, or preprocess the media before the "
                "pipeline. Text-only inputs still work under tracing."
            )
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
        if tf is not None and isinstance(prompts, tf.Tensor):
            if len(prompts.shape) == 0:
                return [prompts.numpy().decode("utf-8")], False
            return [p.numpy().decode("utf-8") for p in prompts], True
        return [
            p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
            for p in prompts
        ], True

    def _carries_media_markers(self, x):
        """True when the prompts still carry media markers.

        Symbolic tensors under `tf.data` tracing cannot be decoded;
        they report no markers so the text-only path stays traceable.
        """
        prompts = x["prompts"] if isinstance(x, dict) else x
        try:
            prompts_list, _ = self._flatten_prompts(prompts)
        except (AttributeError, TypeError):
            prompts_list = []
        return _has_media_markers(
            prompts_list, self.image_token, self.video_token
        )

    def _merge_size(self):
        """Spatial merge factor from whichever converter is configured."""
        converter = self.image_converter or self.video_converter
        return getattr(converter, "merge_size", 2)

    def _media_inputs(self, x):
        """Extract images/videos from the input; empty counts as absent."""
        images = None
        videos = None
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)
        if _empty_media(images):
            images = None
        if _empty_media(videos):
            videos = None
        return images, videos

    def _expand_multimodal(self, x, images, videos, sequence_length):
        """Run converters, expand media markers, check sequence fit."""
        if not self.built:
            self.build(None)
        self._check_converters(images, videos)
        prompts_list, batched = self._flatten_prompts(x["prompts"])
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
        return expanded_sequences, vision_out_images, vision_out_videos, batched

    def _vision_bundle(
        self, token_ids, vision_out_images, vision_out_videos, sgt
    ):
        """``vision_indices``/``position_ids``/concatenated encoder inputs.

        Media are consumed in prompt order across the batch. Returns
        ``None`` pieces when no media is present.
        """
        if vision_out_images is None and vision_out_videos is None:
            return {}
        img_grid = vision_out_images["grid_thw"] if vision_out_images else None
        vid_grid = vision_out_videos["grid_thw"] if vision_out_videos else None
        pixel_values_list = []
        grid_list = []
        if vision_out_images is not None:
            pixel_values_list.append(vision_out_images["pixel_values"])
            grid_list.append(vision_out_images["grid_thw"])
        if vision_out_videos is not None:
            pixel_values_list.append(vision_out_videos["pixel_values"])
            grid_list.append(vision_out_videos["grid_thw"])
        return {
            "vision_indices": self._compute_vision_indices(token_ids),
            "position_ids": self._compute_position_ids(
                token_ids, img_grid, vid_grid, second_per_grid_ts=sgt
            ),
            "pixel_values": ops.concatenate(pixel_values_list, axis=0),
            "image_grid_thw": ops.concatenate(grid_list, axis=0),
        }

    def _process_media(self, images, videos):
        """Run converters and count merged tokens per media item."""
        vision_out_images = (
            self._preprocess_images(images) if images is not None else None
        )
        vision_out_videos = (
            self._preprocess_videos(videos) if videos is not None else None
        )
        merge_size = self._merge_size()

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
        return ops.convert_to_tensor(
            np.concatenate([img_indices, vid_indices]).astype("int32")
        )

    def _temporal_position(self, frame, is_video, media_index, sgt):
        """Temporal position component for a vision token.

        Qwen2-VL uses the raw frame index for all media. Subclasses may
        override to apply time-aware scaling — e.g. Qwen2.5-VL multiplies
        the frame index by ``tokens_per_second * second_per_grid_ts``.

        Args:
            frame: int. Temporal patch index of the token.
            is_video: bool. Whether the token belongs to a video span.
            media_index: int. Index of the media item in its grid.
            sgt: ``second_per_grid_ts`` list from ``_compute_position_ids``,
                or ``None``.
        Returns:
            int temporal offset to add to the span's ``current_pos``.
        """
        return frame

    def _compute_position_ids(
        self, token_ids, image_grid_thw, video_grid_thw, second_per_grid_ts=None
    ):
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
            second_per_grid_ts: optional per-video temporal scale. Unused by
                Qwen2-VL (fixed interval 1); consumed by subclasses via
                ``_temporal_position``.
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

        if second_per_grid_ts is not None:
            # float64 keeps truncation boundaries exact (e.g. 10/3 in
            # float32 truncates a temporal position differently).
            second_per_grid_ts = np.asarray(
                convert_to_numpy(second_per_grid_ts), dtype=np.float64
            )
            if (
                second_per_grid_ts.ndim != 1
                or second_per_grid_ts.shape[0] != video_grid_np.shape[0]
            ):
                raise ValueError(
                    "`second_per_grid_ts` must be a 1-D sequence with one "
                    f"value per video (got {video_grid_np.shape[0]} video "
                    f"grids); received shape {second_per_grid_ts.shape}."
                )

        batch_size, seq_len = token_ids_np.shape
        merge_size = self._merge_size()

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
                        media_index = img_idx
                        img_idx += 1
                    else:
                        t_grid = int(video_grid_np[vid_idx, 0])
                        h_grid = int(video_grid_np[vid_idx, 1])
                        w_grid = int(video_grid_np[vid_idx, 2])
                        media_index = vid_idx
                        vid_idx += 1

                    llm_grid_h = h_grid // merge_size
                    llm_grid_w = w_grid // merge_size
                    frame_tokens = llm_grid_h * llm_grid_w
                    n_tokens = t_grid * frame_tokens

                    span_end = min(i + n_tokens, seq_len)
                    span = span_end - i
                    # h/w indices repeat identically every frame; only the
                    # temporal channel advances per frame.
                    vi = np.arange(span)
                    frames = vi // frame_tokens
                    t_vals = np.array(
                        [
                            self._temporal_position(
                                f, is_video, media_index, second_per_grid_ts
                            )
                            for f in range(t_grid)
                        ]
                    )
                    t_pos[i:span_end] = current_pos + t_vals[frames]
                    h_pos[i:span_end] = (
                        current_pos + (vi // llm_grid_w) % llm_grid_h
                    )
                    w_pos[i:span_end] = current_pos + vi % llm_grid_w

                    current_pos += max(llm_grid_h, llm_grid_w)
                    i = span_end
                else:
                    t_pos[i] = current_pos
                    h_pos[i] = current_pos
                    w_pos[i] = current_pos
                    current_pos += 1
                    i += 1

            nonpad = np.nonzero(ids != self.tokenizer.pad_token_id)[0]
            first_pad = int(nonpad[-1]) + 1 if nonpad.size else 0
            if first_pad > 0 and first_pad < seq_len:
                real_max = int(
                    max(
                        t_pos[:first_pad].max(),
                        h_pos[:first_pad].max(),
                        w_pos[:first_pad].max(),
                    )
                )
                delta = real_max + 1 - int(t_pos[first_pad - 1] + 1)
                if delta > 0:
                    t_pos[first_pad:] += delta
                    h_pos[first_pad:] += delta
                    w_pos[first_pad:] += delta

            all_pos[b, 0] = t_pos
            all_pos[b, 1] = h_pos
            all_pos[b, 2] = w_pos

        return ops.convert_to_tensor(all_pos, dtype="int32")

    def _preprocess_media_items(self, media, leaf_ndim, converter):
        """Convert raw media to patch tensors through ``converter``.

        Media may arrive nested (e.g. per-prompt batching through
        `generate()`), as `tf.RaggedTensor`s, or as dense tensors;
        flatten recursively to leaf arrays — ``(H, W, C)``/``(H, W)``
        for images, ``(T, H, W, C)`` for videos.

        Returns:
            dict with ``pixel_values`` ``(total_patches, T, pH, pW, C)``
            and ``grid_thw`` ``(num_items, 3)``.
        """
        if tf is not None and isinstance(media, tf.RaggedTensor):
            media = media.to_list()
        flat_media = _flatten_media(media, leaf_ndim=leaf_ndim)

        all_patches = []
        all_grid_thw = []
        for item in flat_media:
            is_gray = isinstance(item, np.ndarray) and item.ndim == 2
            if leaf_ndim == 3 and is_gray:
                item = np.stack([item] * 3, axis=-1)
            result = converter(item)
            all_patches.append(ops.convert_to_tensor(result["patches"]))
            all_grid_thw.append(ops.convert_to_tensor(result["grid_thw"]))

        return {
            "pixel_values": ops.concatenate(all_patches, axis=0),
            "grid_thw": ops.concatenate(all_grid_thw, axis=0),
        }

    def _preprocess_images(self, images):
        """Convert raw images to patch tensors using the image converter."""
        return self._preprocess_media_items(images, 3, self.image_converter)

    def _preprocess_videos(self, videos):
        """Convert raw videos to patch tensors using the video converter."""
        converter = self.video_converter or self.image_converter
        return self._preprocess_media_items(videos, 4, converter)

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
        second_per_grid_ts=None,
    ):
        # Popped ahead of input conversion so float64 values are not
        # downcast to float32 during tensorization.
        if second_per_grid_ts is None and isinstance(x, dict):
            second_per_grid_ts = x.get("second_per_grid_ts")
            if second_per_grid_ts is not None:
                x = dict(x)
                x.pop("second_per_grid_ts")
        return self._call(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
            second_per_grid_ts=second_per_grid_ts,
        )

    @preprocessing_function
    def _call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
        second_per_grid_ts=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images, videos = self._media_inputs(x)

        # Text-only fast path: media-free inputs delegate to the base
        # class, whose tokenizer/packer path is safe under `tf.data`
        # tracing. A dict carrying "responses" keeps the eager multimodal
        # path so labels are packed, since the base path cannot. Prompts
        # that still carry media markers after empty-media normalization
        # also take the multimodal path, where `_expand_prompts` raises
        # the descriptive placeholder/media count mismatch.
        if isinstance(x, dict):
            responses_text = x.get("responses", None)
        else:
            responses_text = None
        if images is None and videos is None:
            prompts = x["prompts"] if isinstance(x, dict) else x
            if responses_text is None and not self._carries_media_markers(x):
                return super().call(
                    prompts,
                    y=y,
                    sample_weight=sample_weight,
                    sequence_length=sequence_length,
                )

        (
            expanded_sequences,
            vision_out_images,
            vision_out_videos,
            batched,
        ) = self._expand_multimodal(x, images, videos, sequence_length)

        # Pack prompts (+ responses), shifting for next-token labels.
        if responses_text is not None:
            responses, _ = self._flatten_prompts(responses_text)
            response_ids = [
                self._tokenize_with_special_tokens(r, [], []) for r in responses
            ]
            token_ids, segment_ids = self.multi_packer(
                (expanded_sequences, response_ids),
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
            token_ids, padding_mask = self.packer(
                expanded_sequences,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            x_out = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]

        x_out.update(
            self._vision_bundle(
                x_out["token_ids"],
                vision_out_images,
                vision_out_videos,
                second_per_grid_ts,
            )
        )

        if not batched:
            for key in ("token_ids", "padding_mask", "position_ids"):
                if key in x_out:
                    x_out[key] = ops.squeeze(x_out[key], axis=0)
            y = ops.squeeze(y, axis=0)
            sample_weight = ops.squeeze(sample_weight, axis=0)

        return keras.utils.pack_x_y_sample_weight(x_out, y, sample_weight)

    def generate_preprocess(
        self, x, sequence_length=None, second_per_grid_ts=None
    ):
        # Same pre-conversion capture as `call`.
        if second_per_grid_ts is None and isinstance(x, dict):
            second_per_grid_ts = x.get("second_per_grid_ts")
            if second_per_grid_ts is not None:
                x = dict(x)
                x.pop("second_per_grid_ts")
        return self._generate_preprocess(
            x,
            sequence_length=sequence_length,
            second_per_grid_ts=second_per_grid_ts,
        )

    @preprocessing_function
    def _generate_preprocess(
        self, x, sequence_length=None, second_per_grid_ts=None
    ):
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
        images, videos = self._media_inputs(x)

        # Text-only: delegate to the base class entirely. The base class
        # expects bare strings, so unwrap a `{"prompts": ...}` dict first.
        if images is None and videos is None:
            prompts = x["prompts"] if isinstance(x, dict) else x
            if not self._carries_media_markers(x):
                return super().generate_preprocess(
                    prompts, sequence_length=sequence_length
                )

        sequence_length = sequence_length or self.sequence_length
        (
            expanded_sequences,
            vision_out_images,
            vision_out_videos,
            batched,
        ) = self._expand_multimodal(x, images, videos, sequence_length)

        token_ids, padding_mask = self.packer(
            expanded_sequences,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        x_out = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        x_out.update(
            self._vision_bundle(
                token_ids,
                vision_out_images,
                vision_out_videos,
                second_per_grid_ts,
            )
        )

        if not batched:
            for key in ("token_ids", "padding_mask", "position_ids"):
                if key in x_out:
                    x_out[key] = ops.squeeze(x_out[key], axis=0)
        return x_out

    def _vision_strip_ids(self):
        """Special token ids to strip from generated output."""
        ids_to_strip = list(self.tokenizer.special_token_ids)
        for tok_id in self._special_token_map.values():
            if tok_id not in ids_to_strip:
                ids_to_strip.append(tok_id)
        return ids_to_strip

    def _generate_postprocess_python(self, x):
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids = keras.ops.convert_to_numpy(token_ids).astype("int32")
        padding_mask = keras.ops.convert_to_numpy(padding_mask).astype("bool")

        ids_to_strip = self._vision_strip_ids()

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

        ids_to_strip = self._vision_strip_ids()

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

    @staticmethod
    def _share_image_converter(video_converter, image_converter):
        """Point ``video_converter.image_converter`` at the shared one.

        A deserialized ``Qwen2VLVideoConverter`` carries its own copy of
        the image converter, so mutating the canonical
        ``preprocessor.image_converter`` (e.g. ``max_pixels``) would not
        reach the video path. Rewire when the two configs match, which
        is always true for converters loaded from the same preset.
        """
        if video_converter is None or image_converter is None:
            return
        nested = video_converter.image_converter
        if nested is image_converter:
            return
        if nested.get_config() == image_converter.get_config():
            video_converter.image_converter = image_converter

    @classmethod
    def from_config(cls, config):
        preprocessor = super().from_config(config)
        cls._share_image_converter(
            preprocessor.video_converter, preprocessor.image_converter
        )
        return preprocessor

    @classmethod
    def _add_missing_kwargs(cls, loader, kwargs):
        super()._add_missing_kwargs(loader, kwargs)
        cls._share_image_converter(
            kwargs.get("video_converter"), kwargs.get("image_converter")
        )
