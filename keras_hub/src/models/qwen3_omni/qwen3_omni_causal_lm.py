import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    _vision_indices_to_mask,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm_preprocessor import (
    Qwen3OmniCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3OmniCausalLM")
class Qwen3OmniCausalLM(CausalLM):
    """End-to-end Qwen3-Omni model for causal language modeling.

    Wraps a ``Qwen3OmniBackbone`` (text + optional vision / audio
    encoders) with a reverse-embedding LM head, multimodal M-RoPE
    position-id derivation, and a KV-cached ``generate()`` loop.

    Args:
        backbone: A ``Qwen3OmniBackbone`` instance.
        preprocessor: A ``Qwen3OmniCausalLMPreprocessor`` or ``None``.
    """

    backbone_cls = Qwen3OmniBackbone
    preprocessor_cls = Qwen3OmniCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        self.backbone = backbone
        self.preprocessor = preprocessor
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def _get_token_id(self, attr_name):
        """Return a special-token id from the tokenizer, or ``None``."""
        tokenizer = getattr(
            getattr(self, "preprocessor", None), "tokenizer", None
        )
        return getattr(tokenizer, attr_name, None) if tokenizer else None

    def _compute_initial_embeddings(
        self,
        token_ids,
        audio_features=None,
        audio_indices=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_indices=None,
    ):
        """Build initial embeddings, scattering vision / audio features
        into the text sequence and returning the DeepStack metadata."""
        text_embeds = self.backbone.token_embedding(token_ids)
        backbone = self.backbone

        vision_embeds, deepstack_features = None, None
        if pixel_values is not None and getattr(backbone, "has_vision", False):
            out = backbone.vision_encoder(
                {"pixel_values": pixel_values, "grid_thw": image_grid_thw}
            )
            vision_embeds = out["pooler_output"]
            deepstack_features = out.get("deepstack_features")

        audio_embeds = None
        if audio_features is not None and getattr(backbone, "has_audio", False):
            audio_embeds = backbone.audio_encoder(
                {"input_features": audio_features}
            )

        if getattr(backbone, "is_multimodal", False) and (
            vision_embeds is not None or audio_embeds is not None
        ):
            x = backbone.interleave_embeddings(
                text_embeddings=text_embeds,
                vision_embeddings=vision_embeds,
                vision_indices=vision_indices,
                audio_embeddings=audio_embeds,
                audio_indices=audio_indices,
            )
        else:
            x = text_embeds

        visual_pos_mask = (
            _vision_indices_to_mask(vision_indices, x)
            if vision_indices is not None and deepstack_features is not None
            else None
        )
        return x, deepstack_features, visual_pos_mask

    def compute_multimodal_rope_index(
        self,
        token_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_seqlens=None,
        second_per_grids=None,
        attention_mask=None,
        use_audio_in_video=False,
        image_token_id=None,
        video_token_id=None,
        audio_token_id=None,
        vision_start_token_id=None,
        audio_start_token_id=None,
    ):
        """Compute 3D M-RoPE position IDs for a batch of prompts.

        Vision tokens get ``(temporal, height, width)`` indices on a
        ``spatial_merge_size``-reduced grid; audio tokens advance all
        three channels in lock-step (one slot per compressed audio
        frame); text tokens broadcast a scalar position. Trailing text
        continues from ``max(previous_position) + 1``. With
        ``use_audio_in_video=True`` and a
        ``<vision_start><audio_start>`` prefix, video and audio
        per-step positions are merged in temporal-channel order
        (``bos_len = eos_len = 2``).

        Args:
            token_ids: int array. ``(batch, seq)`` prompt tokens.
            image_grid_thw: int array or None. ``(num_images, 3)``
                ``(t, h, w)`` grid dims in patch units for each image.
            video_grid_thw: int array or None. ``(num_videos, 3)``
                grid dims for each video.
            audio_seqlens: int array or None. ``(num_audios,)`` raw
                audio frame counts (pre-encoder).
            second_per_grids: float array or None. ``(num_videos,)``
                seconds-per-grid-step used to scale temporal positions.
            attention_mask: int array or None. ``(batch, seq)``;
                defaults to all ones.
            use_audio_in_video: bool. Interleave audio and video
                positions when a ``<vision_start><audio_start>`` prefix
                is present.
            image_token_id, video_token_id, audio_token_id,
            vision_start_token_id, audio_start_token_id: int or None.
                Override the tokenizer's default special-token ids.

        Returns:
            Tuple ``(position_ids, deltas)`` of ``int64`` arrays with
            shapes ``(3, batch, seq)`` and ``(batch,)``. Channels
            ``0/1/2`` feed the temporal / height / width M-RoPE
            sub-bands; ``deltas`` is the per-batch offset between the
            max M-RoPE position and the trailing-text cursor.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be 2-D (batch, seq); got {token_ids.shape}"
            )
        batch_size, seq_len = token_ids.shape

        attn_bool = (
            (np.asarray(attention_mask, dtype=np.int64) == 1)
            if attention_mask is not None
            else np.ones_like(token_ids, dtype=bool)
        )

        # Resolve special-token IDs from the tokenizer when not given.
        image_token_id = image_token_id or self._get_token_id("image_token_id")
        video_token_id = video_token_id or self._get_token_id("video_token_id")
        audio_token_id = audio_token_id or self._get_token_id("audio_token_id")
        vision_start_token_id = vision_start_token_id or self._get_token_id(
            "vision_start_token_id"
        )
        audio_start_token_id = audio_start_token_id or self._get_token_id(
            "audio_start_token_id"
        )

        position_id_per_seconds = self.backbone.position_id_per_seconds
        spatial_merge_size = getattr(
            self.backbone.vision_encoder, "spatial_merge_size", 2
        )

        from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
            _get_feat_extract_output_length,
        )

        def _to_np(arr, dtype):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=dtype)
            return None if arr.size == 0 else arr

        image_grid_thw = _to_np(image_grid_thw, np.int64)
        video_grid_thw = _to_np(video_grid_thw, np.int64)
        audio_seqlens = _to_np(audio_seqlens, np.int64)
        second_per_grids = _to_np(second_per_grids, np.float64)

        # Text-only fallback: cumsum-based positions broadcast to 3 channels.
        if (
            image_grid_thw is None
            and video_grid_thw is None
            and audio_seqlens is None
        ):
            cumulative = np.cumsum(attn_bool.astype(np.int64), -1) - 1
            cumulative = np.where(
                attn_bool, cumulative, np.ones_like(cumulative)
            )
            position_ids = np.broadcast_to(
                cumulative[None], (3, batch_size, seq_len)
            ).copy()
            deltas = (
                position_ids.max(axis=0).max(axis=-1)
                + 1
                - attn_bool.sum(axis=-1)
            )
            return position_ids.astype(np.int64), deltas.astype(np.int64)

        position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)
        deltas = np.zeros((batch_size,), dtype=np.int64)
        image_idx = video_idx = audio_idx = 0

        def _vision_pos(start_idx, vision_idx, t_index, grid_hs, grid_ws):
            """``(3, T*llm_h*llm_w)`` M-RoPE positions for one vision clip."""
            llm_h = int(grid_hs[vision_idx]) // spatial_merge_size
            llm_w = int(grid_ws[vision_idx]) // spatial_merge_size
            t_index = np.asarray(t_index, dtype=np.int64)
            t_rep = np.repeat(t_index, llm_h * llm_w)
            h_rep = np.tile(
                np.repeat(np.arange(llm_h, dtype=np.int64), llm_w),
                len(t_index),
            )
            w_rep = np.tile(
                np.arange(llm_w, dtype=np.int64), len(t_index) * llm_h
            )
            return np.stack([t_rep, h_rep, w_rep], axis=0) + start_idx

        def _const(length, start):
            """``(3, length)`` block holding ``arange(length) + start``."""
            return (
                np.broadcast_to(
                    np.arange(length, dtype=np.int64), (3, length)
                ).copy()
                + start
            )

        def _try_index(needle, start):
            try:
                return tokens.index(int(needle), start)
            except ValueError:
                return len(tokens) + 1

        for b in range(batch_size):
            valid = attn_bool[b]
            row = token_ids[b][valid]
            tokens = row.tolist()
            seq_pos_ids = []
            st = 0

            # Count modality occurrences.
            vs_idx = (
                np.where(row == vision_start_token_id)[0]
                if vision_start_token_id is not None
                else np.array([], dtype=np.int64)
            )
            following = (
                row[np.minimum(vs_idx + 1, len(row) - 1)]
                if vs_idx.size
                else np.array([], dtype=row.dtype)
            )
            audio_nums = (
                int(np.sum(row == audio_start_token_id))
                if audio_start_token_id is not None
                else 0
            )
            image_nums = (
                int(np.sum(following == image_token_id))
                if image_token_id is not None
                else 0
            )
            if use_audio_in_video and audio_start_token_id is not None:
                video_nums = int(np.sum(following == audio_start_token_id))
            elif video_token_id is not None:
                video_nums = int(np.sum(following == video_token_id))
            else:
                video_nums = 0

            remain_images, remain_videos, remain_audios = (
                image_nums,
                video_nums,
                audio_nums,
            )
            multimodal_nums = (
                image_nums + audio_nums
                if use_audio_in_video
                else image_nums + video_nums + audio_nums
            )

            for _ in range(multimodal_nums):
                st_idx = int(seq_pos_ids[-1].max() + 1) if seq_pos_ids else 0

                # Locate the next vision_start / audio_start.
                ed_vs = (
                    _try_index(vision_start_token_id, st)
                    if (
                        vision_start_token_id is not None
                        and (remain_videos > 0 or remain_images > 0)
                        and (
                            (
                                image_token_id is not None
                                and image_token_id in tokens
                            )
                            or (
                                video_token_id is not None
                                and video_token_id in tokens
                            )
                        )
                    )
                    else len(tokens) + 1
                )
                ed_as = (
                    _try_index(audio_start_token_id, st)
                    if (
                        audio_start_token_id is not None
                        and audio_token_id is not None
                        and audio_token_id in tokens
                        and remain_audios > 0
                    )
                    else len(tokens) + 1
                )
                min_ed = min(ed_vs, ed_as)

                # Text run preceding the next multimodal marker.
                text_len = min_ed - st
                if text_len > 0:
                    seq_pos_ids.append(_const(text_len, st_idx))
                    st_idx += text_len

                # BOS / EOS lengths (2 each for audio-in-video, else 1).
                aiv = (
                    use_audio_in_video
                    and min_ed == ed_vs
                    and ed_vs + 1 == ed_as
                )
                bos_len = eos_len = 2 if aiv else 1
                seq_pos_ids.append(_const(bos_len, st_idx))
                st_idx += bos_len

                if min_ed == ed_as:
                    # Audio-only branch.
                    if audio_seqlens is None:
                        raise ValueError(
                            "audio_seqlens required for audio markers."
                        )
                    audio_len = int(
                        _get_feat_extract_output_length(
                            int(audio_seqlens[audio_idx])
                        )
                    )
                    seq_pos_ids.append(_const(audio_len, st_idx))
                    st += int(text_len + bos_len + audio_len + eos_len)
                    audio_idx += 1
                    remain_audios -= 1

                elif (
                    min_ed == ed_vs
                    and image_token_id is not None
                    and ed_vs + 1 < len(tokens)
                    and tokens[ed_vs + 1] == image_token_id
                ):
                    # Image-only branch.
                    grid_t = int(image_grid_thw[image_idx][0])
                    t_index = (
                        np.arange(grid_t, dtype=np.int64)
                        * position_id_per_seconds
                    )
                    seq_pos_ids.append(
                        _vision_pos(
                            st_idx,
                            image_idx,
                            t_index,
                            image_grid_thw[:, 1],
                            image_grid_thw[:, 2],
                        )
                    )
                    image_len = int(
                        np.prod(image_grid_thw[image_idx])
                        // (spatial_merge_size**2)
                    )
                    st += int(text_len + bos_len + image_len + eos_len)
                    image_idx += 1
                    remain_images -= 1

                elif (
                    min_ed == ed_vs
                    and not use_audio_in_video
                    and video_token_id is not None
                    and ed_vs + 1 < len(tokens)
                    and tokens[ed_vs + 1] == video_token_id
                ):
                    # Video-only branch.
                    if video_grid_thw is None or second_per_grids is None:
                        raise ValueError(
                            "video_grid_thw and second_per_grids required"
                            " for video markers."
                        )
                    grid_t = int(video_grid_thw[video_idx][0])
                    t_index = (
                        np.arange(grid_t, dtype=np.float64)
                        * float(second_per_grids[video_idx])
                        * position_id_per_seconds
                    ).astype(np.int64)
                    seq_pos_ids.append(
                        _vision_pos(
                            st_idx,
                            video_idx,
                            t_index,
                            video_grid_thw[:, 1],
                            video_grid_thw[:, 2],
                        )
                    )
                    video_len = int(
                        np.prod(video_grid_thw[video_idx])
                        // (spatial_merge_size**2)
                    )
                    st += int(text_len + bos_len + video_len + eos_len)
                    video_idx += 1
                    remain_videos -= 1

                elif aiv:
                    # Audio-in-Video (interleaved).
                    if (
                        video_grid_thw is None
                        or second_per_grids is None
                        or audio_seqlens is None
                    ):
                        raise ValueError(
                            "video_grid_thw, second_per_grids, and"
                            " audio_seqlens required for use_audio_in_video."
                        )
                    audio_len = int(
                        _get_feat_extract_output_length(
                            int(audio_seqlens[audio_idx])
                        )
                    )
                    audio_pos = _const(audio_len, st_idx)
                    grid_t = int(video_grid_thw[video_idx][0])
                    t_index = (
                        np.arange(grid_t, dtype=np.float64)
                        * float(second_per_grids[video_idx])
                        * position_id_per_seconds
                    ).astype(np.int64)
                    video_pos = _vision_pos(
                        st_idx,
                        video_idx,
                        t_index,
                        video_grid_thw[:, 1],
                        video_grid_thw[:, 2],
                    )
                    # Merge sorted by temporal channel.
                    v, a = 0, 0
                    while v < video_pos.shape[-1] and a < audio_pos.shape[-1]:
                        if video_pos[0][v] <= audio_pos[0][a]:
                            seq_pos_ids.append(video_pos[:, v : v + 1])
                            v += 1
                        else:
                            seq_pos_ids.append(audio_pos[:, a : a + 1])
                            a += 1
                    if v < video_pos.shape[-1]:
                        seq_pos_ids.append(video_pos[:, v:])
                    if a < audio_pos.shape[-1]:
                        seq_pos_ids.append(audio_pos[:, a:])
                    video_len = int(
                        np.prod(video_grid_thw[video_idx])
                        // (spatial_merge_size**2)
                    )
                    st += int(
                        text_len + bos_len + audio_len + video_len + eos_len
                    )
                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1
                else:
                    break  # Defensive: avoids infinite loop on bad input.

                st_idx = int(seq_pos_ids[-1].max() + 1)
                seq_pos_ids.append(_const(eos_len, st_idx))

            # Trailing text after the last multimodal block.
            if st < len(tokens):
                st_idx = int(seq_pos_ids[-1].max() + 1) if seq_pos_ids else 0
                seq_pos_ids.append(_const(len(tokens) - st, st_idx))

            flat = (
                np.concatenate(seq_pos_ids, axis=1)
                if seq_pos_ids
                else np.zeros((3, 0), dtype=np.int64)
            )
            full = np.zeros((3, seq_len), dtype=np.int64)
            full[:, valid] = flat
            position_ids[:, b, :] = full
            deltas[b] = int(flat.max() + 1 - len(tokens)) if flat.size else 0
        return position_ids, deltas

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        audio_features=None,
        audio_indices=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_indices=None,
        position_ids=None,
    ):
        """KV-cached forward pass for prompt ingestion and decode.

        On the first call the optional vision / audio encoders run and
        their outputs are scattered into the text embeddings at
        ``vision_indices`` / ``audio_indices``. Subsequent decode steps
        reuse the cache and skip the encoders.

        Args:
            token_ids: int tensor. ``(batch, seq)`` current tokens.
            cache: float tensor. KV cache of shape
                ``(batch, num_layers, 2, max_len, num_kv_heads,
                head_dim)``.
            cache_update_index: int. Position to write the new K/V
                slice into.
            audio_features: float tensor or None. Log-mel input.
            audio_indices: int tensor or None. Flat positions where
                audio embeddings land in the text sequence.
            pixel_values: float tensor or None. Image patches.
            image_grid_thw: int tensor or None. ``(batch, num_images,
                3)`` grid dims.
            vision_indices: int tensor or None. Flat positions where
                vision embeddings land in the text sequence.
            position_ids: int tensor or None. ``(3, batch, seq)``
                M-RoPE positions; if ``None`` they are built from
                ``cache_update_index`` (plus any cached M-RoPE delta)
                and broadcast to all three channels.

        Returns:
            Tuple ``(logits, hidden_states, cache)``. ``logits`` has
            shape ``(batch, seq, vocab)``, ``hidden_states``
            ``(batch, seq, hidden_dim)``, and ``cache`` matches the
            shape of the input cache.
        """
        if audio_features is not None or pixel_values is not None:
            x, deepstack_features, visual_pos_mask = (
                self._compute_initial_embeddings(
                    token_ids,
                    audio_features=audio_features,
                    audio_indices=audio_indices,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    vision_indices=vision_indices,
                )
            )
        else:
            x = self.backbone.token_embedding(token_ids)
            deepstack_features = visual_pos_mask = None

        batch_size = ops.shape(token_ids)[0]
        seq_len = ops.shape(token_ids)[1]
        if position_ids is None:
            positions = ops.arange(seq_len, dtype="int32") + cache_update_index
            positions = ops.repeat(
                ops.expand_dims(positions, 0), batch_size, axis=0
            )
            delta = getattr(self, "_mrope_position_deltas", None)
            if delta is not None:
                positions = positions + ops.expand_dims(
                    ops.cast(delta, positions.dtype), axis=1
                )
            position_ids = ops.stack([positions] * 3, axis=0)

        updated_cache = []
        for i in range(self.backbone.num_layers):
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                position_ids=position_ids,
                cache=cache[:, i, ...],
                cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
            if (
                deepstack_features is not None
                and visual_pos_mask is not None
                and i < len(deepstack_features)
            ):
                x = self.backbone._deepstack_process(
                    x, visual_pos_mask, deepstack_features[i]
                )

        cache = ops.stack(updated_cache, axis=1)
        hidden_states = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        audio_features=None,
        audio_indices=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_indices=None,
        padding_mask=None,
    ):
        """Allocate the KV cache and seed it with the prompt forward pass.

        Caches the per-batch M-RoPE delta on ``self`` when an image grid
        is supplied so trailing-text decode positions stay aligned.
        """
        batch_size = token_ids.shape[0]
        max_length = token_ids.shape[1]
        cache_shape = [
            batch_size,
            self.backbone.num_layers,
            2,
            max_length,
            self.backbone.num_key_value_heads,
            self.backbone.head_dim,
        ]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)

        position_ids = None
        self._mrope_position_deltas = None
        image_token_id = self._get_token_id("image_token_id")
        if image_grid_thw is not None and image_token_id is not None:
            pos_np, deltas_np = self.compute_multimodal_rope_index(
                ops.convert_to_numpy(token_ids),
                image_grid_thw=ops.convert_to_numpy(image_grid_thw),
                attention_mask=(
                    ops.convert_to_numpy(padding_mask)
                    if padding_mask is not None
                    else None
                ),
            )
            position_ids = ops.convert_to_tensor(
                pos_np.astype("int32"), dtype="int32"
            )
            self._mrope_position_deltas = ops.convert_to_tensor(
                deltas_np.astype("int32"), dtype="int32"
            )

        _, hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            0,
            audio_features=audio_features,
            audio_indices=audio_indices,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_indices=vision_indices,
            position_ids=position_ids,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """Compilable per-batch generation step.

        Args:
            inputs: dict. Must contain ``token_ids`` and
                ``padding_mask``; optionally ``audio_features``,
                ``audio_indices``, ``pixel_values``, ``image_grid_thw``,
                ``vision_indices`` for multimodal prompts.
            stop_token_ids: sequence of int or None. Token IDs that
                terminate generation.

        Returns:
            Dict with ``token_ids`` (generated sequence) and
            ``padding_mask`` (True for valid positions, False after
            the first emitted stop token).
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        hidden_states, cache = self._build_cache(
            token_ids,
            audio_features=inputs.get("audio_features"),
            audio_indices=inputs.get("audio_indices"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            vision_indices=inputs.get("vision_indices"),
            padding_mask=padding_mask,
        )
        index = ops.min(ops.sum(ops.cast(padding_mask, "int32"), axis=-1))

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        if stop_token_ids is not None:
            # Mask everything after the first newly-emitted stop token.
            end_locations = ops.cast(
                any_equal(
                    token_ids,
                    stop_token_ids,
                    ops.logical_not(padding_mask),
                ),
                "int32",
            )
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            padding_mask = ops.logical_not(
                ops.cast(cumsum - end_locations, "bool")
            )
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
        audio_features=None,
        audio_indices=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_indices=None,
    ):
        """Score token sequences, with optional multimodal injection.

        Mirrors the embedding path used by ``call_with_cache`` so
        multimodal prompts are scored against the real fused
        representation rather than a text-only fallback.

        Args:
            token_ids: int tensor. ``(batch, seq)`` tokens to score.
            padding_mask: bool tensor or None. ``(batch, seq)``;
                defaults to all ones.
            scoring_mode: ``"logits"`` or ``"loss"``.
            layer_intercept_fn: callable or None. ``fn(x, layer_idx)``
                invoked after each decoder layer (``layer_idx=-1`` is
                the post-embedding hidden state).
            target_ids: int tensor or None. Required when
                ``scoring_mode='loss'``; the ground-truth tokens used
                for per-position cross-entropy.
            audio_features, audio_indices, pixel_values,
            image_grid_thw, vision_indices: see ``call_with_cache``.

        Returns:
            ``(batch, seq, vocab)`` logits when ``scoring_mode='logits'``,
            or ``(batch, seq)`` per-token cross-entropy when
            ``scoring_mode='loss'``.
        """
        if scoring_mode not in ("logits", "loss"):
            raise ValueError("scoring_mode must be 'logits' or 'loss'.")
        if scoring_mode == "loss" and target_ids is None:
            raise ValueError("target_ids is required for scoring_mode='loss'.")

        batch_shape = ops.shape(token_ids)[:2]
        assert len(batch_shape) == 2
        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)
        if layer_intercept_fn is None:
            layer_intercept_fn = lambda x, _i: x  # noqa: E731

        if audio_features is not None or pixel_values is not None:
            x, deepstack_features, visual_pos_mask = (
                self._compute_initial_embeddings(
                    token_ids,
                    audio_features=audio_features,
                    audio_indices=audio_indices,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    vision_indices=vision_indices,
                )
            )
        else:
            x = self.backbone.token_embedding(token_ids)
            deepstack_features = visual_pos_mask = None
        x = layer_intercept_fn(x, -1)

        batch_size = ops.shape(token_ids)[0]
        seq_len = ops.shape(token_ids)[1]
        image_token_id = self._get_token_id("image_token_id")
        if image_grid_thw is not None and image_token_id is not None:
            pos_np, _ = self.compute_multimodal_rope_index(
                ops.convert_to_numpy(token_ids),
                image_grid_thw=ops.convert_to_numpy(image_grid_thw),
                attention_mask=ops.convert_to_numpy(padding_mask),
            )
            position_ids = ops.convert_to_tensor(
                pos_np.astype("int32"), dtype="int32"
            )
        else:
            positions = ops.repeat(
                ops.expand_dims(ops.arange(seq_len, dtype="int32"), 0),
                batch_size,
                axis=0,
            )
            position_ids = ops.stack([positions] * 3, axis=0)

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(
                x,
                position_ids=position_ids,
                decoder_padding_mask=padding_mask,
            )
            if (
                deepstack_features is not None
                and visual_pos_mask is not None
                and i < len(deepstack_features)
            ):
                x = self.backbone._deepstack_process(
                    x, visual_pos_mask, deepstack_features[i]
                )
            x = layer_intercept_fn(x, i)

        logits = self.backbone.token_embedding(
            self.backbone.layer_norm(x), reverse=True
        )
        if scoring_mode == "logits":
            return logits
        return keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )(target_ids, logits)
