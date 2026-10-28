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
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    _get_feat_extract_output_length,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    _vision_indices_to_mask,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_tokenizer import (
    Qwen3OmniTokenizer,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_video_converter import (
    Qwen3OmniVideoConverter,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


def _merge_vision_outputs(outputs):
    """Merge per-item converter outputs into flat batched tensors.

    Each converter call returns ``patches`` ``(N_i, T, P, P, C)`` and
    ``grid_thw`` ``(3,)`` for a single image or video. The backbone's
    functional graph consumes them unbatched: all patches concatenated
    on axis 0 and one ``(n_items, 3)`` grid.
    """
    if len(outputs) == 1:
        out = outputs[0]
        return {
            "patches": out["patches"],
            "grid_thw": ops.expand_dims(out["grid_thw"], 0),
        }
    return {
        "patches": ops.concatenate([out["patches"] for out in outputs], axis=0),
        "grid_thw": ops.stack([out["grid_thw"] for out in outputs]),
    }


def _merge_patches_and_grids(image_out, video_out):
    """Merge image / video converter outputs for the shared encoder.

    Returns ``(pixel_values, grid_thw)`` as flat concatenated tensors in
    ``[images..., videos...]`` order, or ``(None, None)`` when neither
    modality is present.
    """
    patches, grids = [], []
    for out in (image_out, video_out):
        if out is not None:
            patches.append(out["patches"])
            grids.append(out["grid_thw"])
    if not patches:
        return None, None
    pixel_values = (
        ops.concatenate(patches, axis=0) if len(patches) > 1 else patches[0]
    )
    grid_thw = ops.concatenate(grids, axis=0) if len(grids) > 1 else grids[0]
    return pixel_values, grid_thw


@keras_hub_export(
    "keras_hub.models.Qwen3OmniCausalLMPreprocessor",
)
class Qwen3OmniCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3-Omni causal LM preprocessor with vision and audio support.

    For text-only usage this preprocessor behaves identically to the
    base ``CausalLMPreprocessor``. When converters are attached and the
    user supplies multimodal inputs as a dict, this preprocessor:

    1. Converts images, videos, and audio to feature tensors via the
       attached converters.
    2. Replaces ``<|image_pad|>`` / ``<|video_pad|>`` / ``<|audio_pad|>``
       placeholder tokens in the text with the correct number of
       repeated placeholders (one per visual / audio token produced by
       the corresponding encoder).
    3. Computes flat ``vision_indices`` and ``audio_indices`` describing
       where the encoder outputs should be scattered into the text
       embedding sequence.

    The resulting dict matches the functional graph of
    ``Qwen3OmniBackbone`` (text + vision + audio inputs) so it can flow
    through ``model.fit``, ``model.predict``, ``model.generate``, or
    ``Qwen3OmniCausalLM.score`` without further reshaping.

    Args:
        tokenizer: A ``Qwen3OmniTokenizer`` instance. Special-token IDs
            (``image_token_id``, ``video_token_id``, etc.) are resolved
            from the tokenizer.
        audio_converter: A ``Qwen3OmniAudioConverter`` instance, or
            ``None``.
        image_converter: A ``Qwen3OmniImageConverter`` instance, or
            ``None``.
        video_converter: A ``Qwen3OmniVideoConverter`` instance, or
            ``None``.
        sequence_length: int. Total padded sequence length. Default
            1024.
        add_start_token: bool. Whether to prepend the start token.
            Default ``True``.
        add_end_token: bool. Whether to append the end token. Default
            ``True``.
    """

    backbone_cls = Qwen3OmniBackbone
    tokenizer_cls = Qwen3OmniTokenizer
    audio_converter_cls = Qwen3OmniAudioConverter
    image_converter_cls = Qwen3OmniImageConverter
    video_converter_cls = Qwen3OmniVideoConverter

    _SPECIAL_TOKEN_ATTRS = [
        "im_start_token",
        "end_token",
        "vision_start_token",
        "vision_end_token",
        "image_token",
        "video_token",
        "audio_start_token",
        "audio_end_token",
        "audio_token",
    ]

    def __init__(
        self,
        tokenizer,
        audio_converter=None,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=True,
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
        self.audio_converter = audio_converter
        self.image_converter = image_converter
        self.video_converter = video_converter

        self._cached_special_token_map = None
        self._cached_special_token_pattern = None

    @property
    def image_token_id(self):
        return getattr(self.tokenizer, "image_token_id", None)

    @property
    def video_token_id(self):
        return getattr(self.tokenizer, "video_token_id", None)

    @property
    def audio_token_id(self):
        return getattr(self.tokenizer, "audio_token_id", None)

    @property
    def image_token(self):
        return getattr(self.tokenizer, "image_token", None)

    @property
    def video_token(self):
        return getattr(self.tokenizer, "video_token", None)

    @property
    def audio_token(self):
        return getattr(self.tokenizer, "audio_token", None)

    @property
    def _special_token_map(self):
        """Lazily resolve a ``token_string -> token_id`` map.

        We read special tokens from the attached tokenizer so that
        callers cannot accidentally desync the IDs between tokenizer
        and preprocessor.
        """
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
        if self._cached_special_token_pattern is None:
            mapping = self._special_token_map
            if not mapping:
                self._cached_special_token_pattern = re.compile(r"$^")
            else:
                self._cached_special_token_pattern = re.compile(
                    "(" + "|".join(re.escape(t) for t in mapping) + ")"
                )
        return self._cached_special_token_pattern

    def build(self, input_shape):
        super().build(input_shape)
        start_value = []
        if (
            self.add_start_token
            and getattr(self.tokenizer, "im_start_token_id", None) is not None
        ):
            start_value = [self.tokenizer.im_start_token_id]
        self.multi_packer = MultiSegmentPacker(
            start_value=start_value,
            end_value=self.tokenizer.end_token_id or [],
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )

    @staticmethod
    def _as_item_list(value, batched_rank):
        """Normalize a modality input to a list of per-item tensors.

        ``convert_preprocessing_inputs`` collapses a list of same-shape
        arrays into a single stacked tensor, so multi-item inputs can
        arrive as a Python list, a ``tf.RaggedTensor``, or a dense
        tensor with a leading item axis. ``batched_rank`` is the rank
        of a stacked batch (image 4, video 5, audio 2).
        """
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, tf.RaggedTensor):
            # Rows of a singly-ragged tensor are already dense; only
            # deeper raggedness needs ``to_tensor``.
            return [
                row.to_tensor() if isinstance(row, tf.RaggedTensor) else row
                for row in value
            ]
        if value is not None and len(ops.shape(value)) == batched_rank:
            return [value[i] for i in range(int(ops.shape(value)[0]))]
        return [value]

    def _process_multimodal_inputs(self, x):
        """Run modality-specific converters on a sample dict.

        Returns a tuple of per-modality outputs. Image and video
        outputs are kept separate (their grids and pixel buffers are
        not concatenated) because the text token placeholders, the
        encoder calls, and the M-RoPE positions all require the per-
        modality identity to be preserved.
        """
        audio_features, audio_lens = None, None
        if "audio" in x and self.audio_converter is not None:
            feats = []
            for a in self._as_item_list(x["audio"], 2):
                f = self.audio_converter(a)
                # A rank-1 waveform returns squeezed ``(frames, mels)``;
                # the backbone consumes ``(n_audios, frames, mels)``.
                if len(ops.shape(f)) == 2:
                    f = ops.expand_dims(f, 0)
                feats.append(f)
            # HF pads the batch to the longest clip and tracks true
            # lengths through ``feature_attention_mask``; we keep the
            # per-item frame counts in ``audio_lens`` instead.
            audio_lens = [int(ops.shape(f)[-2]) for f in feats]
            max_len = max(audio_lens)
            feats = [
                ops.pad(
                    f,
                    [
                        [0, 0],
                        [0, max_len - int(ops.shape(f)[-2])],
                        [0, 0],
                    ],
                    mode="constant",
                )
                for f in feats
            ]
            audio_features = (
                ops.concatenate(feats, axis=0) if len(feats) > 1 else feats[0]
            )

        image_out = None
        if "images" in x and self.image_converter is not None:
            image_out = _merge_vision_outputs(
                [
                    self.image_converter(img)
                    for img in self._as_item_list(x["images"], 4)
                ]
            )

        video_out = None
        if "video" in x and self.video_converter is not None:
            video_out = _merge_vision_outputs(
                [
                    self.video_converter(vid)
                    for vid in self._as_item_list(x["video"], 5)
                ]
            )

        return audio_features, audio_lens, image_out, video_out

    def _spatial_merge_size(self):
        """Resolve the spatial merge factor from the attached converter."""
        return getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

    def _position_id_per_seconds(self):
        return getattr(
            getattr(self, "backbone", None), "position_id_per_seconds", 25
        )

    def _has_multimodal_converters(self):
        return (
            self.image_converter is not None
            or self.video_converter is not None
            or self.audio_converter is not None
        )

    def _compute_position_ids(
        self,
        token_ids,
        padding_mask,
        image_grid_thw,
        video_grid_thw,
        audio_lens,
        use_audio_in_video=False,
    ):
        """NumPy M-RoPE positions ``(batch, 3, seq)`` + per-batch deltas."""
        from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm import (
            _compute_mrope_index,
        )

        audio_seqlens = (
            None
            if audio_lens is None
            else np.asarray(audio_lens, dtype=np.int64)
        )
        n_videos = (
            0
            if video_grid_thw is None
            else int(ops.convert_to_numpy(video_grid_thw).shape[0])
        )
        second_per_grids = (
            np.ones(n_videos, dtype=np.float64) if n_videos else None
        )
        pos_np, deltas_np = _compute_mrope_index(
            ops.convert_to_numpy(token_ids),
            image_grid_thw=(
                None
                if image_grid_thw is None
                else ops.convert_to_numpy(image_grid_thw)
            ),
            video_grid_thw=(
                None
                if video_grid_thw is None
                else ops.convert_to_numpy(video_grid_thw)
            ),
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
            attention_mask=ops.convert_to_numpy(padding_mask),
            use_audio_in_video=use_audio_in_video,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            vision_start_token_id=getattr(
                self.tokenizer, "vision_start_token_id", None
            ),
            audio_start_token_id=getattr(
                self.tokenizer, "audio_start_token_id", None
            ),
            position_id_per_seconds=self._position_id_per_seconds(),
            spatial_merge_size=self._spatial_merge_size(),
            audio_chunk_len=self._audio_chunk_len(),
        )
        return np.transpose(pos_np, (1, 0, 2)).astype("int32"), deltas_np

    def _num_image_tokens(self, image_grid_thw):
        """Return per-image visual token counts."""
        if image_grid_thw is None:
            return []
        merge = self._spatial_merge_size()
        grid_np = ops.convert_to_numpy(image_grid_thw)
        if grid_np.ndim == 1:
            grid_np = grid_np[None, :]
        return [
            int(grid_np[i, 0])
            * (int(grid_np[i, 1]) // merge)
            * (int(grid_np[i, 2]) // merge)
            for i in range(grid_np.shape[0])
        ]

    def _audio_chunk_len(self):
        """Mel frames per audio-encoder chunk (``n_window * 2``)."""
        ae = getattr(getattr(self, "backbone", None), "audio_encoder", None)
        return int(getattr(ae, "n_window", 50)) * 2

    def _num_audio_tokens(self, audio_lens):
        """Return per-audio post-CNN token counts.

        ``audio_lens`` is the list of true mel-frame counts per clip
        (``feature_attention_mask.sum(-1)`` in HF terms).
        """
        if not audio_lens:
            return []
        chunk_len = self._audio_chunk_len()
        return [
            int(_get_feat_extract_output_length(int(n), chunk_len))
            for n in audio_lens
        ]

    def _tokenize_with_special_tokens(
        self,
        text,
        num_image_tokens,
        num_video_tokens,
        num_audio_tokens,
    ):
        """Tokenize ``text`` while expanding multimodal placeholders.

        The BPE tokenizer may decompose multi-codepoint specials into
        sub-pieces, which would corrupt the placeholder count. To avoid
        that, we split the input by every known special token, tokenize
        only the surrounding text segments, and emit special-token IDs
        directly. Image / video / audio placeholders are repeated by
        their per-instance count.
        """
        parts = self._special_token_pattern.split(text)
        token_map = self._special_token_map
        image_token = self.image_token
        video_token = self.video_token
        audio_token = self.audio_token

        all_ids = []
        img_idx = vid_idx = aud_idx = 0
        for part in parts:
            if not part:
                continue
            if part in token_map:
                if image_token is not None and part == image_token:
                    n = (
                        num_image_tokens[img_idx]
                        if img_idx < len(num_image_tokens)
                        else 1
                    )
                    img_idx += 1
                    all_ids.extend([token_map[part]] * n)
                elif video_token is not None and part == video_token:
                    n = (
                        num_video_tokens[vid_idx]
                        if vid_idx < len(num_video_tokens)
                        else 1
                    )
                    vid_idx += 1
                    all_ids.extend([token_map[part]] * n)
                elif audio_token is not None and part == audio_token:
                    n = (
                        num_audio_tokens[aud_idx]
                        if aud_idx < len(num_audio_tokens)
                        else 1
                    )
                    aud_idx += 1
                    all_ids.extend([token_map[part]] * n)
                else:
                    all_ids.append(token_map[part])
            else:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    def _compute_indices(self, token_ids, target_token_ids):
        """Return flat int32 indices into ``(batch * seq_len)`` matching
        any of ``target_token_ids``.

        Returns an empty tensor when ``target_token_ids`` is empty or
        when no positions match. The output is a 1-D tensor (the
        functional graph reshapes per-batch shapes as needed).
        """
        if not target_token_ids:
            return tf.zeros((0,), dtype="int32")
        token_ids_np = ops.convert_to_numpy(token_ids).reshape(-1)
        mask = np.zeros_like(token_ids_np, dtype=bool)
        for tid in target_token_ids:
            if tid is None:
                continue
            mask |= token_ids_np == tid
        return tf.constant(np.where(mask)[0].astype(np.int32))

    def _flatten_prompts(self, prompts):
        """Flatten a string / Tensor / list of prompts to a list of str."""
        if isinstance(prompts, str):
            return [prompts], False
        if isinstance(prompts, tf.Tensor):
            if len(prompts.shape) == 0:
                return [prompts.numpy().decode("utf-8")], False
            return [p.numpy().decode("utf-8") for p in prompts], True
        if isinstance(prompts, (list, tuple)):
            return [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ], True
        return [str(prompts)], False

    def _add_multimodal_to_output(
        self,
        output,
        audio_features,
        audio_indices,
        image_pixel_values,
        image_grid_thw,
        vision_indices,
        audio_lens=None,
        backbone=None,
    ):
        """Write all multimodal keys into ``output``.

        Present modalities receive their real tensors; absent modalities
        receive zero-length placeholder tensors with the correct dtype
        so Keras input-spec validation passes cleanly.
        """
        # Resolve encoder attributes for static placeholder shapes.
        # The attached converters are the primary source: a preprocessor
        # loaded from a preset always carries them, while `backbone` is
        # only available when a caller wires it in explicitly.
        ve = getattr(backbone, "vision_encoder", None) if backbone else None
        ae = getattr(backbone, "audio_encoder", None) if backbone else None
        conv_img = self.image_converter or self.video_converter
        conv_aud = self.audio_converter
        has_vision = (
            image_pixel_values is not None
            or conv_img is not None
            or ve is not None
            or (getattr(backbone, "has_vision", False) if backbone else False)
        )
        has_audio = (
            audio_features is not None
            or conv_aud is not None
            or ae is not None
            or (getattr(backbone, "has_audio", False) if backbone else False)
        )

        if has_vision:
            dim_src = ve if ve is not None else conv_img
            if image_pixel_values is not None:
                output["pixel_values"] = image_pixel_values
            elif dim_src is not None:
                output["pixel_values"] = np.zeros(
                    (
                        0,
                        dim_src.temporal_patch_size,
                        dim_src.patch_size,
                        dim_src.patch_size,
                        getattr(dim_src, "in_channels", 3),
                    ),
                    dtype="float32",
                )
            output["image_grid_thw"] = (
                image_grid_thw
                if image_grid_thw is not None
                else np.zeros((0, 3), dtype="int32")
            )
            output["vision_indices"] = (
                vision_indices
                if vision_indices is not None
                else np.zeros((0,), dtype="int32")
            )

        if has_audio:
            mel_src = ae if ae is not None else conv_aud
            if audio_features is not None:
                output["audio_features"] = audio_features
            elif mel_src is not None:
                num_mels = getattr(
                    mel_src, "num_mel_bins", getattr(mel_src, "num_mels", 128)
                )
                output["audio_features"] = np.zeros(
                    (0, 0, num_mels), dtype="float32"
                )
            output["audio_indices"] = (
                audio_indices
                if audio_indices is not None
                else np.zeros((0,), dtype="int32")
            )
            output["audio_lengths"] = (
                np.asarray(audio_lens, dtype="int32")
                if audio_lens
                else np.zeros((0,), dtype="int32")
            )

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
            x = self.tokenizer(x)
            token_ids, padding_mask = self.packer(
                x,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            self._add_multimodal_to_output(
                x,
                None,
                None,
                None,
                None,
                None,
                backbone=getattr(self, "backbone", None),
            )
            if self._has_multimodal_converters():
                x["position_ids"], _ = self._compute_position_ids(
                    x["token_ids"], x["padding_mask"], None, None, None
                )
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        # Multimodal dict input.
        prompts, _ = self._flatten_prompts(x["prompts"])
        responses_text = x.get("responses", None)
        use_audio_in_video = bool(x.get("use_audio_in_video", False))
        (
            audio_features,
            audio_lens,
            image_out,
            video_out,
        ) = self._process_multimodal_inputs(x)

        image_grid_thw = (
            image_out["grid_thw"] if image_out is not None else None
        )
        video_grid_thw = (
            video_out["grid_thw"] if video_out is not None else None
        )
        num_image_tokens = self._num_image_tokens(image_grid_thw)
        num_video_tokens = self._num_image_tokens(video_grid_thw)
        num_audio_tokens = self._num_audio_tokens(audio_lens)

        prompt_ids = [
            self._tokenize_with_special_tokens(
                p, num_image_tokens, num_video_tokens, num_audio_tokens
            )
            for p in prompts
        ]

        if responses_text is not None:
            responses, _ = self._flatten_prompts(responses_text)
            response_ids = [
                self._tokenize_with_special_tokens(r, [], [], [])
                for r in responses
            ]
            prompt_ragged = tf.ragged.constant(prompt_ids, dtype="int32")
            response_ragged = tf.ragged.constant(response_ids, dtype="int32")
            token_ids, segment_ids = self.multi_packer(
                (prompt_ragged, response_ragged),
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            padding_mask = token_ids != self.tokenizer.pad_token_id
            response_mask = segment_ids == 1
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            y = token_ids[..., 1:]
            sample_weight = response_mask[..., 1:]
        else:
            ragged = tf.ragged.constant(prompt_ids, dtype="int32")
            token_ids, padding_mask = self.packer(
                ragged,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]

        # Compute flat indices for vision + audio scatter inputs.
        vision_indices = self._compute_indices(
            x["token_ids"],
            [self.image_token_id, self.video_token_id],
        )
        audio_indices = self._compute_indices(
            x["token_ids"], [self.audio_token_id]
        )

        pixel_values, merged_grid_thw = _merge_patches_and_grids(
            image_out, video_out
        )
        pos_ids, _ = self._compute_position_ids(
            x["token_ids"],
            x["padding_mask"],
            image_grid_thw,
            video_grid_thw,
            audio_lens,
            use_audio_in_video=use_audio_in_video,
        )
        self._add_multimodal_to_output(
            x,
            audio_features,
            audio_indices,
            pixel_values,
            merged_grid_thw,
            vision_indices,
            audio_lens=audio_lens,
            backbone=getattr(self, "backbone", None),
        )
        if self._has_multimodal_converters():
            x["position_ids"] = pos_ids
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)

        if not isinstance(x, dict):
            x = self.tokenizer(x)
            token_ids, padding_mask = self.packer(
                x, sequence_length=sequence_length, add_end_value=False
            )
            result = {"token_ids": token_ids, "padding_mask": padding_mask}
            self._add_multimodal_to_output(
                result,
                None,
                None,
                None,
                None,
                None,
                backbone=getattr(self, "backbone", None),
            )
            if self._has_multimodal_converters():
                result["position_ids"], _ = self._compute_position_ids(
                    token_ids, padding_mask, None, None, None
                )
            return result

        prompts, _ = self._flatten_prompts(x["prompts"])
        (
            audio_features,
            audio_lens,
            image_out,
            video_out,
        ) = self._process_multimodal_inputs(x)
        image_grid_thw = (
            image_out["grid_thw"] if image_out is not None else None
        )
        video_grid_thw = (
            video_out["grid_thw"] if video_out is not None else None
        )
        num_image_tokens = self._num_image_tokens(image_grid_thw)
        num_video_tokens = self._num_image_tokens(video_grid_thw)
        num_audio_tokens = self._num_audio_tokens(audio_lens)

        prompt_ids = [
            self._tokenize_with_special_tokens(
                p, num_image_tokens, num_video_tokens, num_audio_tokens
            )
            for p in prompts
        ]
        ragged = tf.ragged.constant(prompt_ids, dtype="int32")
        token_ids, padding_mask = self.packer(
            ragged,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        vision_indices = self._compute_indices(
            token_ids, [self.image_token_id, self.video_token_id]
        )
        audio_indices = self._compute_indices(token_ids, [self.audio_token_id])

        pixel_values, merged_grid_thw = _merge_patches_and_grids(
            image_out, video_out
        )
        pos_ids, _ = self._compute_position_ids(
            token_ids,
            padding_mask,
            image_grid_thw,
            video_grid_thw,
            audio_lens,
            use_audio_in_video=bool(x.get("use_audio_in_video", False)),
        )
        result = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        self._add_multimodal_to_output(
            result,
            audio_features,
            audio_indices,
            pixel_values,
            merged_grid_thw,
            vision_indices,
            audio_lens=audio_lens,
            backbone=getattr(self, "backbone", None),
        )
        if self._has_multimodal_converters():
            result["position_ids"] = pos_ids
        self._add_initial_embeddings(result)
        return result

    def _add_initial_embeddings(self, result):
        """Fold vision / audio encoder outputs into text embeddings.

        The encoders contain eager-only Python (per-item grid loops,
        dynamic shapes), which cannot run inside ``generate_step``'s
        ``tf.function``. ``generate_preprocess`` runs eagerly, so we run
        the encoders here and hand the model a ready-made
        ``initial_embeddings`` tensor (plus ``deepstack_features`` and
        ``visual_pos_mask`` when vision features exist). Requires the
        owning ``Qwen3OmniCausalLM`` to have wired ``self.backbone``.
        """
        backbone = getattr(self, "backbone", None)
        if backbone is None:
            return
        pixel_values = result.get("pixel_values")
        audio_features = result.get("audio_features")
        has_pixels = (
            pixel_values is not None and int(ops.shape(pixel_values)[0]) > 0
        )
        has_audio = (
            audio_features is not None and int(ops.shape(audio_features)[0]) > 0
        )
        if not has_pixels and not has_audio:
            return

        text_embeddings = backbone.token_embedding(result["token_ids"])
        vision_embeddings = deepstack_features = None
        if has_pixels and getattr(backbone, "has_vision", False):
            vision_out = backbone.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "grid_thw": result["image_grid_thw"],
                }
            )
            vision_embeddings = vision_out["pooler_output"]
            deepstack_features = vision_out.get("deepstack_features")
        audio_embeddings = None
        if has_audio and getattr(backbone, "has_audio", False):
            audio_embeddings = backbone.audio_encoder(
                {
                    "input_features": audio_features,
                    "feature_lengths": result.get("audio_lengths"),
                }
            )

        result["initial_embeddings"] = backbone.interleave_embeddings(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            vision_indices=result.get("vision_indices"),
            audio_embeddings=audio_embeddings,
            audio_indices=result.get("audio_indices"),
        )
        if deepstack_features:
            # Stack into one tensor: a list value would break the
            # Functional input mapping when the dict is passed to
            # `model()`, and `deepstack_features[i]` works the same on
            # the stacked tensor.
            result["deepstack_features"] = ops.stack(deepstack_features, axis=0)
            result["visual_pos_mask"] = _vision_indices_to_mask(
                result["vision_indices"], result["initial_embeddings"]
            )
