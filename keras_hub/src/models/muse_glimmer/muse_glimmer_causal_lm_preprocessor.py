import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.utils.tensor_utils import assert_tf_installed
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.MuseGlimmerCausalLMPreprocessor")
class MuseGlimmerCausalLMPreprocessor(CausalLMPreprocessor):
    """MuseGlimmer causal LM preprocessor with optional image/video inputs.

    For text-only usage this behaves identically to the base
    `CausalLMPreprocessor`. When an `image_converter`/`video_converter` is
    provided and the input contains `"images"`/`"videos"`, each occurrence
    of the tokenizer's `image_token_id`/`video_token_id` placeholder in the
    prompt is expanded to the number of merged vision tokens for that
    image/frame, and flat `vision_indices` are computed for scattering the
    vision encoder's output into the text embedding sequence. MuseGlimmer
    uses standard sequential text positions throughout — no M-RoPE is
    needed, since spatial awareness lives entirely inside the vision tower.

    Args:
        tokenizer: A `MuseGlimmerTokenizer` instance.
        image_converter: A `MuseGlimmerImageConverter` instance, or `None`.
        video_converter: A `MuseGlimmerVideoConverter` instance, or `None`.
        sequence_length: int. The length of the packed inputs.
        add_start_token: bool. Whether to prepend the tokenizer's start
            token. Defaults to `True`.
        add_end_token: bool. Whether to append the tokenizer's end token.
            Defaults to `True`.
    """

    backbone_cls = MuseGlimmerBackbone
    tokenizer_cls = MuseGlimmerTokenizer
    image_converter_cls = MuseGlimmerImageConverter
    video_converter_cls = MuseGlimmerVideoConverter

    def __init__(
        self,
        tokenizer,
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
        self.image_converter = image_converter
        self.video_converter = video_converter

    def _compute_vision_indices(self, token_ids):
        token_ids_np = np.asarray(token_ids)
        img_mask = (token_ids_np == self.tokenizer.image_token_id).reshape(-1)
        vid_mask = (token_ids_np == self.tokenizer.video_token_id).reshape(-1)
        img_indices = np.where(img_mask)[0].astype(np.int32)
        vid_indices = np.where(vid_mask)[0].astype(np.int32)
        return tf.constant(np.concatenate([img_indices, vid_indices], axis=0))

    def _expand_vision_placeholders(
        self, ids, num_image_tokens, num_video_tokens
    ):
        expanded = []
        img_idx, vid_idx = 0, 0
        for tok in ids:
            if tok == self.tokenizer.image_token_id:
                n = (
                    num_image_tokens[img_idx]
                    if img_idx < len(num_image_tokens)
                    else 1
                )
                expanded.extend([self.tokenizer.image_token_id] * n)
                img_idx += 1
            elif tok == self.tokenizer.video_token_id:
                n = (
                    num_video_tokens[vid_idx]
                    if vid_idx < len(num_video_tokens)
                    else 1
                )
                expanded.extend([self.tokenizer.video_token_id] * n)
                vid_idx += 1
            else:
                expanded.append(tok)
        return expanded

    def _num_merged_tokens(self, grid_thw, merge_size):
        counts = []
        for t, h, w in np.asarray(grid_thw):
            t, h, w = int(t), int(h), int(w)
            counts.append(t * (h // merge_size) * (w // merge_size))
        return counts

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        images, videos = None, None
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)

        if images is None and videos is None:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        assert_tf_installed(
            "MuseGlimmerCausalLMPreprocessor with images or videos"
        )
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts = x["prompts"]
        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]

        pixel_values_list, grid_list = [], []
        num_image_tokens, num_video_tokens = [], []

        if images is not None and self.image_converter is not None:
            flat_images = (
                images if isinstance(images, (list, tuple)) else [images]
            )
            for img in flat_images:
                result = self.image_converter(img)
                pixel_values_list.append(tf.constant(result["patches"]))
                grid_list.append(tf.constant(result["grid_thw"]))
            image_grids = np.stack(
                [np.asarray(g) for g in grid_list[: len(flat_images)]]
            )
            num_image_tokens = self._num_merged_tokens(
                image_grids, self.image_converter.merge_size
            )

        if videos is not None and self.video_converter is not None:
            flat_videos = (
                videos if isinstance(videos, (list, tuple)) else [videos]
            )
            video_grid_start = len(grid_list)
            for vid in flat_videos:
                result = self.video_converter(vid)
                pixel_values_list.append(tf.constant(result["patches"]))
                grid_list.append(tf.constant(result["grid_thw"]))
            video_grids = np.stack(
                [np.asarray(g) for g in grid_list[video_grid_start:]]
            )
            num_video_tokens = self._num_merged_tokens(
                video_grids, self.video_converter.merge_size
            )

        prompts_list = (
            [
                p.numpy().decode("utf-8") if hasattr(p, "numpy") else str(p)
                for p in prompts
            ]
            if isinstance(prompts, (tf.Tensor, list, tuple))
            else [str(prompts)]
        )

        expanded_sequences = []
        for prompt_str in prompts_list:
            ids = self.tokenizer(prompt_str)
            ids = ids.numpy().tolist() if hasattr(ids, "numpy") else list(ids)
            ids = self._expand_vision_placeholders(
                ids, num_image_tokens, num_video_tokens
            )
            expanded_sequences.append(ids)

        token_ids_ragged = tf.ragged.constant(expanded_sequences, dtype="int32")
        token_ids, padding_mask = self.packer(
            token_ids_ragged,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        vision_indices = self._compute_vision_indices(token_ids)
        if pixel_values_list:
            combined_pixel_values = tf.concat(pixel_values_list, axis=0)
            combined_grid_thw = tf.concat(grid_list, axis=0)
        else:
            combined_pixel_values = tf.zeros((0, 0), dtype="float32")
            combined_grid_thw = tf.zeros((0, 3), dtype="int32")

        return {
            "token_ids": token_ids if batched else tf.squeeze(token_ids, 0),
            "padding_mask": padding_mask
            if batched
            else tf.squeeze(padding_mask, 0),
            "pixel_values": combined_pixel_values,
            "image_grid_thw": combined_grid_thw,
            "vision_indices": vision_indices,
        }

    def get_config(self):
        config = super().get_config()
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        if self.video_converter is not None:
            config["video_converter"] = keras.layers.serialize(
                self.video_converter
            )
        return config
