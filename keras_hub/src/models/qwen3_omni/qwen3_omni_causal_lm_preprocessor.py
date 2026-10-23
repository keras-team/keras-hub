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

    def _process_multimodal_inputs(self, x):
        """Run modality-specific converters on a sample dict.

        Returns a tuple of per-modality outputs. Image and video
        outputs are kept separate (their grids and pixel buffers are
        not concatenated) because the text token placeholders, the
        encoder calls, and the M-RoPE positions all require the per-
        modality identity to be preserved.
        """
        audio_features = None
        if "audio" in x and self.audio_converter is not None:
            audio_features = self.audio_converter(x["audio"])

        image_out = None
        if "images" in x and self.image_converter is not None:
            image_out = self.image_converter(x["images"])

        video_out = None
        if "video" in x and self.video_converter is not None:
            video_out = self.video_converter(x["video"])

        return audio_features, image_out, video_out

    def _spatial_merge_size(self):
        """Resolve the spatial merge factor from the attached converter."""
        return getattr(
            self.image_converter or self.video_converter,
            "spatial_merge_size",
            2,
        )

    def _num_image_tokens(self, image_grid_thw):
        """Return per-image visual token counts."""
        if image_grid_thw is None:
            return []
        merge = self._spatial_merge_size()
        grid_np = (
            image_grid_thw.numpy()
            if hasattr(image_grid_thw, "numpy")
            else np.asarray(image_grid_thw)
        )
        if grid_np.ndim == 1:
            grid_np = grid_np[None, :]
        return [
            int(grid_np[i, 0])
            * (int(grid_np[i, 1]) // merge)
            * (int(grid_np[i, 2]) // merge)
            for i in range(grid_np.shape[0])
        ]

    def _num_audio_tokens(self, audio_features):
        """Return per-audio post-CNN token counts.

        ``audio_features`` is expected to be ``(num_audios, time,
        num_mel_bins)`` from the converter; we treat the time axis as
        the input length to the encoder downsampler.
        """
        if audio_features is None:
            return []
        feat_np = (
            audio_features.numpy()
            if hasattr(audio_features, "numpy")
            else np.asarray(audio_features)
        )
        if feat_np.ndim == 2:
            feat_np = feat_np[None, ...]
        counts = []
        for i in range(feat_np.shape[0]):
            time_len = int(feat_np[i].shape[0])
            counts.append(int(_get_feat_extract_output_length(time_len)))
        return counts

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
        backbone=None,
    ):
        """Write all multimodal keys into ``output``.

        Present modalities receive their real tensors; absent modalities
        receive zero-length placeholder tensors with the correct dtype
        so Keras input-spec validation passes cleanly.
        """
        # Resolve encoder attributes for static placeholder shapes.
        ve = getattr(backbone, "vision_encoder", None) if backbone else None
        ae = getattr(backbone, "audio_encoder", None) if backbone else None
        has_vision = (
            getattr(backbone, "has_vision", False) if backbone else False
        ) or image_pixel_values is not None
        has_audio = (
            getattr(backbone, "has_audio", False) if backbone else False
        ) or audio_features is not None

        if has_vision:
            if image_pixel_values is not None:
                output["pixel_values"] = image_pixel_values
            elif ve is not None:
                output["pixel_values"] = np.zeros(
                    (
                        0,
                        0,
                        ve.temporal_patch_size,
                        ve.patch_size,
                        ve.patch_size,
                        ve.in_channels,
                    ),
                    dtype="float32",
                )
            output["image_grid_thw"] = (
                image_grid_thw
                if image_grid_thw is not None
                else np.zeros((0, 0, 3), dtype="int32")
            )
            output["vision_indices"] = (
                vision_indices
                if vision_indices is not None
                else np.zeros((0, 0), dtype="int32")
            )

        if has_audio:
            if audio_features is not None:
                output["audio_features"] = audio_features
            elif ae is not None:
                output["audio_features"] = np.zeros(
                    (0, 0, ae.num_mel_bins), dtype="float32"
                )
            output["audio_indices"] = (
                audio_indices
                if audio_indices is not None
                else np.zeros((0, 0), dtype="int32")
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
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        # Multimodal dict input.
        prompts, _ = self._flatten_prompts(x["prompts"])
        responses_text = x.get("responses", None)
        audio_features, image_out, video_out = self._process_multimodal_inputs(
            x
        )

        image_grid_thw = (
            image_out["grid_thw"] if image_out is not None else None
        )
        video_grid_thw = (
            video_out["grid_thw"] if video_out is not None else None
        )
        num_image_tokens = self._num_image_tokens(image_grid_thw)
        num_video_tokens = self._num_image_tokens(video_grid_thw)
        num_audio_tokens = self._num_audio_tokens(audio_features)

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

        image_pixel_values = (
            image_out["patches"] if image_out is not None else None
        )
        self._add_multimodal_to_output(
            x,
            audio_features,
            audio_indices,
            image_pixel_values,
            image_grid_thw,
            vision_indices,
            backbone=getattr(self, "backbone", None),
        )
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
            return {"token_ids": token_ids, "padding_mask": padding_mask}

        prompts, _ = self._flatten_prompts(x["prompts"])
        audio_features, image_out, video_out = self._process_multimodal_inputs(
            x
        )
        image_grid_thw = (
            image_out["grid_thw"] if image_out is not None else None
        )
        video_grid_thw = (
            video_out["grid_thw"] if video_out is not None else None
        )
        num_image_tokens = self._num_image_tokens(image_grid_thw)
        num_video_tokens = self._num_image_tokens(video_grid_thw)
        num_audio_tokens = self._num_audio_tokens(audio_features)

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

        image_pixel_values = (
            image_out["patches"] if image_out is not None else None
        )
        result = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        self._add_multimodal_to_output(
            result,
            audio_features,
            audio_indices,
            image_pixel_values,
            image_grid_thw,
            vision_indices,
            backbone=getattr(self, "backbone", None),
        )
        return result
