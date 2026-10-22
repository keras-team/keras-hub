import re
from unittest import mock

import grain
import keras
import numpy as np
import pytest

from keras_hub.src.models.qwen3_vl import qwen3_vl_causal_lm_preprocessor
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3VLTokenizer
from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import (
    Qwen3VLVideoConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.tokenizers.byte_pair_tokenizer import bytes_to_unicode
from keras_hub.src.utils.tensor_utils import tf

try:
    import torch
    import torchvision  # noqa: F401
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers import models
    from tokenizers import pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers import Qwen2VLImageProcessor
    from transformers import Qwen3VLConfig
    from transformers import Qwen3VLProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )
    from transformers.video_utils import VideoMetadata
except ImportError:
    torch = None


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
]
IMAGE_PROMPT = "air <|vision_start|><|image_pad|><|vision_end|> at"
VIDEO_PROMPT = "air <|vision_start|><|video_pad|><|vision_end|> at"
# HF transformers 5.x replaces only the pad token with the video blocks.
BARE_VIDEO_PROMPT = "air <|video_pad|> at"


def build_vocabulary():
    """Byte-level vocabulary, so that any ASCII text can be tokenized."""
    merges = [("Ġ", "a"), ("Ġ", "t"), ("a", "i"), ("ai", "r"), ("Ġa", "t")]
    vocabulary = {}
    for char in bytes_to_unicode()[1]:
        vocabulary[char] = len(vocabulary)
    for left, right in merges:
        vocabulary[left + right] = len(vocabulary)
    for token in SPECIAL_TOKENS:
        vocabulary[token] = len(vocabulary)
    return vocabulary, [f"{left} {right}" for left, right in merges]


class Qwen3VLCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocabulary, self.merges = build_vocabulary()
        self.tokenizer = Qwen3VLTokenizer(
            vocabulary=self.vocabulary, merges=self.merges
        )
        # 2 pixel patches, 2x2 merge: a 8x8 image is a 4x4 patch grid.
        converter_kwargs = dict(
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=16,
            max_pixels=1000,
        )
        self.image_converter = Qwen3VLImageConverter(**converter_kwargs)
        self.video_converter = Qwen3VLVideoConverter(
            fps=2.0, min_frames=2, max_frames=8, **converter_kwargs
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "video_converter": self.video_converter,
            "sequence_length": 64,
        }
        self.preprocessor = Qwen3VLCausalLMPreprocessor(**self.init_kwargs)
        rng = np.random.RandomState(0)
        self.image = rng.randint(0, 255, (8, 8, 3)).astype("uint8")
        self.video = rng.randint(0, 255, (6, 8, 8, 3)).astype("uint8")

    def tok(self, name):
        return self.vocabulary[name]

    def test_preprocessor_basics(self):
        tokens = self.tokenizer("airplane at airport").tolist()
        padded = tokens + [self.tok("<|endoftext|>")] * (8 - len(tokens))
        end = self.tok("<|im_end|>")
        self.run_preprocessor_test(
            cls=Qwen3VLCausalLMPreprocessor,
            init_kwargs={**self.init_kwargs, "sequence_length": 8},
            input_data=["airplane at airport"],
            expected_output=(
                {
                    "token_ids": [padded[:7] + [padded[7]]],
                    "padding_mask": [[1] * 8],
                    "position_ids": [[[i] * 3 for i in range(8)]],
                    "pixel_values": np.zeros((1, 0, 2, 2, 2, 3), "float32"),
                    "image_grid_thw": np.zeros((1, 0, 3), "int32"),
                    "vision_indices": np.zeros((1, 0), "int32"),
                },
                [padded[1:8] + [end]],
                [[1] * 8],
            ),
        )

    def test_empty_prompts_with_media_raise(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            preprocessor.generate_preprocess(
                {"prompts": [], "images": []}, sequence_length=16
            )

    def test_text_position_ids_skip_padding(self):
        x = self.preprocessor.generate_preprocess("air")
        length = len(self.tokenizer("air"))
        self.assertAllEqual(x["padding_mask"][:length], [True] * length)
        expected = np.zeros((64, 3), "int32")
        expected[:length] = np.arange(length)[:, None]
        self.assertAllEqual(x["position_ids"], expected)

    def test_start_and_end_tokens(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            **self.init_kwargs, add_start_token=True
        )
        # The Qwen tokenizer has no start token, so nothing is prepended.
        x, y, sw = preprocessor(["air at"] * 2)
        end = self.tok("<|im_end|>")
        length = len(self.tokenizer("air at"))
        self.assertEqual(x["token_ids"].shape, (2, 64))
        self.assertEqual(int(y[0, length - 1]), end)
        self.assertEqual(int(keras.ops.convert_to_numpy(sw[0]).sum()), length)

    def test_generate_postprocess(self):
        x = self.preprocessor.generate_preprocess(["airplane at airport"])
        output = self.preprocessor.generate_postprocess(x)
        self.assertEqual(output[0], "airplane at airport")

    def test_prompts_dict_without_media_has_empty_vision_inputs(self):
        x = self.preprocessor.generate_preprocess({"prompts": ["air", "at"]})
        self.assertEqual(x["pixel_values"].shape, (2, 0, 2, 2, 2, 3))
        self.assertEqual(x["vision_indices"].shape, (2, 0))
        self.assertEqual(x["position_ids"].shape, (2, 64, 3))

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_text_path_in_tf_data(self):
        ds = tf.data.Dataset.from_tensor_slices(["air at", "air"]).batch(2)
        x = next(iter(ds.map(self.preprocessor.generate_preprocess)))
        self.assertEqual(x["position_ids"].shape, (2, 64, 3))
        length = len(self.tokenizer("air at"))
        self.assertAllEqual(x["position_ids"][0, :length, 0], range(length))
        ds = tf.data.Dataset.from_tensor_slices({"prompts": ["air at", "air"]})
        x, y, sw = next(iter(ds.batch(2).map(self.preprocessor)))
        self.assertEqual(x["position_ids"].shape, (2, 64, 3))
        self.assertEqual(y.shape, (2, 64))

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_media_path_in_tf_data(self):
        cases = [
            ("images", IMAGE_PROMPT, self.image, {}),
            ("videos", VIDEO_PROMPT, self.video, {"video_fps": 4.0}),
        ]
        for key, prompt, media, extra in cases:
            sample = {"prompts": prompt, key: media, **extra}
            batch = {k: [v, v] for k, v in sample.items()}
            # Each sample of the dataset matches the eager call.
            ds = tf.data.Dataset.from_tensors(sample)
            expected = self.preprocessor.generate_preprocess(sample)
            x = next(iter(ds.map(self.preprocessor.generate_preprocess)))
            self.assertAllClose(x, expected)
            expected = self.preprocessor(sample)
            self.assertAllClose(next(iter(ds.map(self.preprocessor))), expected)
            # So does a batch of samples.
            ds = tf.data.Dataset.from_tensor_slices(batch).batch(2)
            expected = self.preprocessor(batch)
            self.assertAllClose(next(iter(ds.map(self.preprocessor))), expected)

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_video_fps_list_in_tf_data(self):
        sample = {
            "prompts": VIDEO_PROMPT + VIDEO_PROMPT,
            "videos": [self.video, self.video],
            "video_fps": [4.0, 8.0],
        }
        expected = self.preprocessor.generate_preprocess(sample)
        ds = tf.data.Dataset.from_tensors(sample)
        x = next(iter(ds.map(self.preprocessor.generate_preprocess)))
        self.assertAllClose(x, expected)

    def test_video_fps_list_of_tensors(self):
        sample = {
            "prompts": VIDEO_PROMPT + VIDEO_PROMPT,
            "videos": [self.video, self.video],
            "video_fps": [4.0, 8.0],
        }
        expected = self.preprocessor.generate_preprocess(sample)
        to_numpy = qwen3_vl_causal_lm_preprocessor.convert_to_numpy

        def host_only(x):
            # GPU tensors only reach the host when passed one by one.
            if isinstance(x, (list, tuple)):
                raise TypeError("can't convert a list of device tensors")
            return to_numpy(x)

        fps = [
            keras.ops.convert_to_tensor(4.0),
            keras.ops.convert_to_tensor(8.0),
        ]
        for video_fps in (fps, tuple(fps)):
            sample["video_fps"] = video_fps
            with mock.patch.object(
                qwen3_vl_causal_lm_preprocessor, "convert_to_numpy", host_only
            ):
                x = self.preprocessor.generate_preprocess(sample)
            self.assertAllClose(x, expected)

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_media_prompts_as_bytes_and_string_tensors(self):
        expected = self.preprocessor.generate_preprocess(
            {"prompts": [IMAGE_PROMPT] * 2, "images": [self.image] * 2}
        )
        # As `generate` sees them from a `tf.data.Dataset`.
        for prompts in [
            [IMAGE_PROMPT.encode()] * 2,
            np.array([IMAGE_PROMPT.encode()] * 2),
            tf.constant([IMAGE_PROMPT] * 2),
        ]:
            x = self.preprocessor.generate_preprocess(
                {"prompts": prompts, "images": [self.image] * 2}
            )
            self.assertAllClose(x, expected)

    def test_media_outputs_are_assembled_on_host(self):
        x = self.preprocessor._preprocess_media(
            {"prompts": [IMAGE_PROMPT], "images": [self.image]},
            sequence_length=64,
            training=False,
        )
        expected = {
            "token_ids": "int32",
            "padding_mask": "bool",
            "position_ids": "int32",
            "pixel_values": "float32",
            "image_grid_thw": "int32",
            "vision_indices": "int32",
        }
        for key, dtype in expected.items():
            self.assertIsInstance(x[key], np.ndarray)
            self.assertEqual(x[key].dtype, dtype)

    def test_text_path_in_grain(self):
        length = len(self.tokenizer("air at"))
        # Grain maps with several threads, which would race to build the
        # packer lazily.
        self.preprocessor.build(None)
        ds = grain.MapDataset.source(["air at", "air"])
        for x in ds.map(self.preprocessor.generate_preprocess):
            assert_grain_safe_types(x)
            self.assertEqual(x["position_ids"].shape, (64, 3))
        ds = grain.MapDataset.source([{"prompts": "air at"}])
        x, y, sw = next(iter(ds.map(self.preprocessor)))
        assert_grain_safe_types((x, y, sw))
        self.assertAllEqual(x["position_ids"][:length, 0], range(length))
        self.assertEqual(y.shape, (64,))

    def test_generate_rounds_vision_shapes_up(self):
        prompt = " ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * 5)
        x = self.preprocessor.generate_preprocess(
            {"prompts": [prompt], "images": [[self.image] * 5]}
        )
        # 5 images of 4 tokens round up to 24 tokens and 6 grid rows.
        self.assertEqual(x["pixel_values"].shape[1], 24 * 4)
        self.assertEqual(x["image_grid_thw"].shape, (1, 6, 3))
        indices = keras.ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(indices.shape, (1, 24))
        mask = keras.ops.convert_to_numpy(x["padding_mask"])[0]
        self.assertFalse(mask[indices[0, 20:]].any())
        # Training keeps the exact shapes.
        x, _, _ = self.preprocessor(
            {"prompts": [prompt], "images": [[self.image] * 5]}
        )
        self.assertEqual(x["pixel_values"].shape[1], 20 * 4)

    def test_grayscale_and_rgba_images(self):
        def preprocess(image):
            x = self.preprocessor.generate_preprocess(
                {"prompts": IMAGE_PROMPT, "images": image}
            )
            return keras.ops.convert_to_numpy(x["pixel_values"])

        gray = self.image[..., 0]
        rgb = preprocess([np.repeat(gray[..., None], 3, axis=-1)])
        # One 2D array is a single grayscale image, as is `(h, w, 1)`.
        self.assertAllClose(preprocess(gray), rgb)
        self.assertAllClose(preprocess([gray[..., None]]), rgb)
        # Alpha is dropped, as PIL `convert("RGB")` does.
        alpha = np.full(self.image.shape[:2] + (1,), 7, "uint8")
        self.assertAllClose(
            preprocess([np.concatenate([self.image, alpha], axis=-1)]),
            preprocess([self.image]),
        )

    def test_image(self):
        x = self.preprocessor.generate_preprocess(
            {"prompts": IMAGE_PROMPT, "images": [self.image]}
        )
        image_id = self.tok("<|image_pad|>")
        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        # A 4x4 patch grid merges into 2x2 = 4 tokens.
        self.assertEqual(int((token_ids == image_id).sum()), 4)
        start = int(np.flatnonzero(token_ids == image_id)[0])
        self.assertAllEqual(x["vision_indices"], np.arange(start, start + 4))
        self.assertAllEqual(x["image_grid_thw"], [[1, 4, 4]])
        # Without video, the encoder repeats the single frame itself.
        self.assertEqual(x["pixel_values"].shape, (16, 1, 2, 2, 3))
        positions = keras.ops.convert_to_numpy(x["position_ids"])
        # Text "air " is 4 tokens, so the image starts at position 5 (after
        # the vision start token) and spans max(2, 2) positions.
        before = start
        self.assertAllEqual(positions[start : start + 4, 0], [before] * 4)
        self.assertAllEqual(
            positions[start : start + 4, 1],
            [before, before, before + 1, before + 1],
        )
        self.assertAllEqual(
            positions[start : start + 4, 2],
            [before, before + 1, before, before + 1],
        )
        # Text resumes at start + max(h, w) // merge, not after every token.
        self.assertEqual(positions[start + 4, 0], before + 2)

    def test_video_expands_per_temporal_patch(self):
        # 8 frames at 8 fps sampled at 2 fps keep 2 frames: indices 0 and 7.
        # One chunk, timestamp (0 + 7 / 8) / 2 = 0.4375 -> 0.4.
        video = np.concatenate([self.video, self.video[:2]])
        x = self.preprocessor.generate_preprocess(
            {"prompts": VIDEO_PROMPT, "videos": [video], "video_fps": [8.0]}
        )
        text = self.tokenizer.detokenize(
            keras.ops.convert_to_numpy(x["token_ids"])[
                keras.ops.convert_to_numpy(x["padding_mask"])
            ]
        )
        self.assertIn("<0.4 seconds><|vision_start|>", text)
        self.assertEqual(text.count("<|video_pad|>"), 4)
        # The chunk block plus the template's own marker, like HF.
        self.assertEqual(text.count("<|vision_start|>"), 2)

    def test_media_sequence_length_as_tensor(self):
        x = {"prompts": IMAGE_PROMPT, "images": [self.image]}
        expected = self.preprocessor.generate_preprocess(x, sequence_length=32)
        output = self.preprocessor.generate_preprocess(
            x, sequence_length=keras.ops.convert_to_tensor(32)
        )
        self.assertAllEqual(output["token_ids"], expected["token_ids"])

    def test_video_fps_count_must_match_videos(self):
        x = {
            "prompts": VIDEO_PROMPT + VIDEO_PROMPT,
            "videos": [self.video, self.video],
            "video_fps": [8.0],
        }
        with self.assertRaisesRegex(ValueError, "one entry per video"):
            self.preprocessor.generate_preprocess(x)

    def test_video_chunks_and_timestamps(self):
        # 12 frames at 6 fps with a 2 fps target keep 4 frames:
        # indices 0, 4, 7, 11 -> chunk times (0 + 4) / 12 and (7 + 11) / 12.
        video = np.repeat(self.video, 2, axis=0)
        x = self.preprocessor.generate_preprocess(
            {"prompts": VIDEO_PROMPT, "videos": [video], "video_fps": 6.0}
        )
        ids = keras.ops.convert_to_numpy(x["token_ids"])[
            keras.ops.convert_to_numpy(x["padding_mask"])
        ]
        text = self.tokenizer.detokenize(ids)
        self.assertIn("<0.3 seconds><|vision_start|>", text)
        self.assertIn("<1.5 seconds><|vision_start|>", text)
        self.assertEqual(text.count("<|video_pad|>"), 8)
        # One grid row holds both chunks, and both chunks are in the stream.
        self.assertAllEqual(x["image_grid_thw"], [[2, 4, 4]])
        positions = keras.ops.convert_to_numpy(x["position_ids"])
        pads = np.flatnonzero(ids == self.tok("<|video_pad|>"))
        first, second = pads[:4], pads[4:]
        # T is constant inside a chunk and text advances between chunks.
        self.assertEqual(len(set(positions[first, 0])), 1)
        self.assertLess(positions[first[0], 0], positions[second[0], 0])

    def test_wrapped_video_marker_keeps_template_tokens(self):
        outputs = [
            self.preprocessor.generate_preprocess(
                {"prompts": prompt, "videos": [self.video]}
            )
            for prompt in (VIDEO_PROMPT, BARE_VIDEO_PROMPT)
        ]
        counts = []
        for x in outputs:
            ids = keras.ops.convert_to_numpy(x["token_ids"])[
                keras.ops.convert_to_numpy(x["padding_mask"])
            ]
            counts.append(
                [
                    int(np.sum(ids == self.tok(token)))
                    for token in ("<|vision_start|>", "<|vision_end|>")
                ]
            )
        # Like HF, the template's own markers stay around the video blocks.
        self.assertEqual(counts[0], [counts[1][0] + 1, counts[1][1] + 1])

    def test_mixed_image_and_video_batch_is_padded(self):
        prompts = [IMAGE_PROMPT + " " + VIDEO_PROMPT, "air"]
        x = self.preprocessor.generate_preprocess(
            {
                "prompts": prompts,
                "images": [[self.image], None],
                "videos": [[self.video], []],
            }
        )
        # Sample 0: one image (16 patches) and one video chunk (16 patches).
        self.assertEqual(x["pixel_values"].shape, (2, 32, 2, 2, 2, 3))
        self.assertAllEqual(
            x["image_grid_thw"], [[[1, 4, 4], [1, 4, 4]], [[0, 0, 0]] * 2]
        )
        self.assertEqual(
            keras.ops.convert_to_numpy(x["pixel_values"])[1].any(), False
        )
        indices = keras.ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(indices.shape, (2, 8))
        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        types = np.isin(
            token_ids,
            [self.tok("<|image_pad|>"), self.tok("<|video_pad|>")],
        )
        # Real indices are the vision tokens in prompt order.
        self.assertAllEqual(indices[0], np.flatnonzero(types[0]))
        # Padded indices land on padding positions of their own sample.
        mask = keras.ops.convert_to_numpy(x["padding_mask"])
        self.assertFalse(mask[1][indices[1]].any())

    def test_image_features_scatter_into_vision_tokens(self):
        x = self.preprocessor.generate_preprocess(
            {
                "prompts": [IMAGE_PROMPT, IMAGE_PROMPT],
                "images": [[self.image]] * 2,
            }
        )
        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        indices = keras.ops.convert_to_numpy(x["vision_indices"])
        # Indices are positions within each sample.
        vision = np.take_along_axis(token_ids, indices, axis=1)
        self.assertTrue((vision == self.tok("<|image_pad|>")).all())
        self.assertAllEqual(indices[0], indices[1])
        self.assertEqual(len(set(indices[0].tolist())), 4)

    def test_batch_without_media_has_empty_vision_inputs(self):
        x = self.preprocessor.generate_preprocess(
            {"prompts": ["air", "at"], "images": [None, []]}
        )
        self.assertEqual(x["pixel_values"].shape, (2, 0, 2, 2, 2, 3))
        self.assertEqual(x["image_grid_thw"].shape, (2, 0, 3))
        self.assertEqual(x["vision_indices"].shape, (2, 0))
        self.assertEqual(x["position_ids"].shape, (2, 64, 3))

    def test_single_prompt_drops_batch_axis(self):
        x = self.preprocessor.generate_preprocess(
            {"prompts": IMAGE_PROMPT, "images": self.image}
        )
        self.assertEqual(x["token_ids"].shape, (64,))
        # Without video, the encoder repeats the single frame itself.
        self.assertEqual(x["pixel_values"].shape, (16, 1, 2, 2, 3))
        self.assertEqual(x["image_grid_thw"].shape, (1, 3))
        self.assertEqual(x["vision_indices"].shape, (4,))

    def test_call_with_media_returns_labels(self):
        x, y, sw = self.preprocessor(
            {
                "prompts": [IMAGE_PROMPT, "air"],
                "images": [[self.image], None],
            }
        )
        self.assertEqual(x["token_ids"].shape, (2, 64))
        self.assertEqual(x["position_ids"].shape, (2, 64, 3))
        self.assertEqual(x["vision_indices"].shape, (2, 4))
        length = int(keras.ops.convert_to_numpy(x["padding_mask"])[0].sum())
        self.assertAllEqual(
            keras.ops.convert_to_numpy(y)[0, : length - 1],
            keras.ops.convert_to_numpy(x["token_ids"])[0, 1:length],
        )
        self.assertEqual(
            int(keras.ops.convert_to_numpy(y)[0, length - 2]),
            self.tok("<|im_end|>"),
        )
        # The 4 image placeholder labels carry no weight.
        sw = keras.ops.convert_to_numpy(sw)
        self.assertEqual(int(sw[0].sum()), length - 1 - 4)
        y = keras.ops.convert_to_numpy(y)
        self.assertFalse(sw[y == self.tok("<|image_pad|>")].any())

    def test_narrow_grayscale_image_is_one_image(self):
        # A width of 3 must not read as the channel axis.
        gray = self.image[:, :3, 0]
        x = self.preprocessor.generate_preprocess(
            {"prompts": IMAGE_PROMPT, "images": gray}
        )
        rgb = self.preprocessor.generate_preprocess(
            {
                "prompts": IMAGE_PROMPT,
                "images": [np.repeat(gray[..., None], 3, axis=-1)],
            }
        )
        self.assertAllClose(x["pixel_values"], rgb["pixel_values"])

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_ragged_images(self):
        small = self.image[:4]
        ragged = tf.ragged.constant([[self.image, small]], ragged_rank=2)
        prompt = IMAGE_PROMPT + IMAGE_PROMPT
        x = self.preprocessor.generate_preprocess(
            {"prompts": [prompt], "images": ragged}
        )
        expected = self.preprocessor.generate_preprocess(
            {"prompts": [prompt], "images": [[self.image, small]]}
        )
        self.assertAllClose(x["pixel_values"], expected["pixel_values"])
        self.assertAllEqual(x["image_grid_thw"], expected["image_grid_thw"])

    def test_timestamps_use_first_and_last_frame(self):
        converter = Qwen3VLVideoConverter(
            patch_size=2, temporal_patch_size=4, spatial_merge_size=2
        )
        preprocessor = Qwen3VLCausalLMPreprocessor(
            self.tokenizer, video_converter=converter
        )
        # HF takes (first + last) / 2, not the mean of all four frames.
        self.assertEqual(preprocessor._timestamps([0, 1, 2, 7], 1.0), [3.5])

    def test_unknown_keys_raise(self):
        with self.assertRaisesRegex(ValueError, "responses"):
            self.preprocessor.generate_preprocess(
                {"prompts": "air", "responses": "at"}
            )

    def test_video_converter_only(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            self.tokenizer,
            video_converter=self.video_converter,
            sequence_length=64,
        )
        x = preprocessor.generate_preprocess(
            {"prompts": VIDEO_PROMPT, "videos": [self.video]}
        )
        expected = self.preprocessor.generate_preprocess(
            {"prompts": VIDEO_PROMPT, "videos": [self.video]}
        )
        for key in expected:
            self.assertAllClose(x[key], expected[key])
        # Text-only inputs still get empty vision inputs.
        x = preprocessor.generate_preprocess("air at")
        self.assertEqual(x["pixel_values"].shape, (0, 2, 2, 2, 3))

    def test_media_path_in_grain(self):
        self.preprocessor.build(None)
        sample = {"prompts": IMAGE_PROMPT, "images": [self.image]}
        expected = self.preprocessor.generate_preprocess(sample)
        ds = grain.MapDataset.source([sample, sample])
        for x in ds.map(self.preprocessor.generate_preprocess):
            assert_grain_safe_types(x)
            for key in expected:
                self.assertAllClose(x[key], expected[key])
        video = {"prompts": VIDEO_PROMPT, "videos": [self.video]}
        x, y, sw = next(
            iter(grain.MapDataset.source([video]).map(self.preprocessor))
        )
        assert_grain_safe_types((x, y, sw))
        self.assertEqual(y.shape, (64,))

    def test_media_without_converter_raises(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(self.tokenizer)
        with self.assertRaisesRegex(ValueError, "image_converter"):
            preprocessor.generate_preprocess(
                {"prompts": IMAGE_PROMPT, "images": [self.image]}
            )
        with self.assertRaisesRegex(ValueError, "video_converter"):
            preprocessor({"prompts": "air", "videos": [self.video]})

    def test_marker_and_media_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "more image markers"):
            self.preprocessor.generate_preprocess(
                {"prompts": IMAGE_PROMPT, "videos": [self.video]}
            )
        with self.assertRaisesRegex(ValueError, "fewer image or video"):
            self.preprocessor.generate_preprocess(
                {"prompts": "air", "images": [self.image]}
            )
        with self.assertRaisesRegex(ValueError, "one entry per prompt"):
            self.preprocessor.generate_preprocess(
                {"prompts": ["air", "at"], "images": [[self.image]]}
            )

    def test_adjacent_image_markers_raise(self):
        prompt = "air <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        with self.assertRaisesRegex(ValueError, "no text between them"):
            self.preprocessor.generate_preprocess(
                {"prompts": prompt, "images": [self.image, self.image]}
            )

    def test_truncated_vision_tokens_raise(self):
        with self.assertRaisesRegex(ValueError, "do not fit"):
            self.preprocessor.generate_preprocess(
                {"prompts": IMAGE_PROMPT, "images": [self.image]},
                sequence_length=5,
            )

    def test_full_sample_cannot_take_padded_vision_indices(self):
        # Sample 1 has no vision tokens and fills the whole sequence, so the
        # padded vision indices have nowhere to go.
        with self.assertRaisesRegex(
            ValueError, "Sample 1 has no padding position"
        ):
            self.preprocessor.generate_preprocess(
                {
                    "prompts": [IMAGE_PROMPT, "air" * 20],
                    "images": [[self.image], None],
                },
                sequence_length=16,
            )

    def test_config_round_trip(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            **self.init_kwargs, default_video_fps=30.0, add_end_token=False
        )
        config = preprocessor.get_config()
        restored = Qwen3VLCausalLMPreprocessor.from_config(dict(config))
        self.assertEqual(restored.get_config(), config)
        self.assertEqual(restored.default_video_fps, 30.0)
        self.assertFalse(restored.add_end_token)
        self.assertIsInstance(restored.image_converter, Qwen3VLImageConverter)
        self.assertIsInstance(restored.video_converter, Qwen3VLVideoConverter)
        self.assertEqual(restored.video_converter.patch_size, 2)

    def test_default_video_fps_is_used(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            **self.init_kwargs, default_video_fps=6.0
        )
        x = preprocessor.generate_preprocess(
            {"prompts": VIDEO_PROMPT, "videos": [self.video]}
        )
        text = self.tokenizer.detokenize(
            keras.ops.convert_to_numpy(x["token_ids"])[
                keras.ops.convert_to_numpy(x["padding_mask"])
            ]
        )
        # Indices 0 and 5 at 6 fps.
        self.assertIn("<0.4 seconds>", text)

    def test_outputs_run_through_backbone(self):
        vision_encoder = Qwen3VLVisionEncoder(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=16,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=(0, 1),
        )
        backbone = Qwen3VLBackbone(
            vocabulary_size=len(self.vocabulary),
            num_layers=3,
            num_query_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            hidden_dim=16,
            intermediate_dim=32,
            mrope_section=(2, 1, 1),
            vision_encoder=vision_encoder,
        )
        x = self.preprocessor.generate_preprocess(
            {
                "prompts": [IMAGE_PROMPT + " " + VIDEO_PROMPT, "air"],
                "images": [[self.image], None],
                "videos": [[self.video], None],
            },
            sequence_length=48,
        )
        output = backbone(x)
        self.assertEqual(tuple(output.shape), (2, 48, 16))
        self.assertTrue(np.isfinite(keras.ops.convert_to_numpy(output)).all())

    def build_hf_reference(self):
        """HF processor and model that share this test's tokenizer."""
        merges = [tuple(m.split(" ")) for m in self.merges]
        plain = {
            k: v for k, v in self.vocabulary.items() if k not in SPECIAL_TOKENS
        }
        backend = Tokenizer(models.BPE(vocab=plain, merges=merges))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.add_special_tokens(SPECIAL_TOKENS)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
        )
        shared = dict(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 16, "longest_edge": 1000},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        )
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(**shared),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(
                fps=2, min_frames=2, max_frames=8, **shared
            ),
        )
        config = Qwen3VLConfig(
            text_config=dict(
                vocab_size=len(self.vocabulary),
                hidden_size=8,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=8,
            ),
            vision_config=dict(
                depth=1,
                hidden_size=8,
                num_heads=1,
                intermediate_size=8,
                out_hidden_size=8,
                patch_size=2,
                spatial_merge_size=2,
                deepstack_visual_indexes=[0],
            ),
        )
        return processor, Qwen3VLModel(config)

    def assert_matches_hf(self, prompts, images, videos, video_fps):
        processor, model = self.build_hf_reference()
        # HF takes pre-sampled frames together with their source indices.
        sampled, metadata = [], []
        for per_sample, fps_list in zip(videos, video_fps):
            for video, fps in zip(per_sample or [], fps_list or []):
                indices = keras.ops.convert_to_numpy(
                    self.video_converter.sample_frame_indices(len(video), fps)
                )
                sampled.append(video[indices])
                metadata.append(
                    VideoMetadata(
                        total_num_frames=len(video),
                        fps=fps,
                        frames_indices=indices.tolist(),
                    )
                )
        flat_images = [i for per_sample in images for i in per_sample or []]
        kwargs = {"padding": True, "return_tensors": "pt"}
        if flat_images:
            kwargs["images"] = flat_images
        if sampled:
            kwargs["videos"] = sampled
            kwargs["videos_kwargs"] = {
                "video_metadata": metadata,
                "do_sample_frames": False,
            }
        hf = processor(text=list(prompts), **kwargs)
        positions, _ = model.get_rope_index(
            hf["input_ids"],
            hf["mm_token_type_ids"],
            image_grid_thw=hf.get("image_grid_thw"),
            video_grid_thw=hf.get("video_grid_thw"),
            attention_mask=hf["attention_mask"],
        )

        x = self.preprocessor.generate_preprocess(
            {
                "prompts": list(prompts),
                "images": images,
                "videos": videos,
                "video_fps": video_fps,
            },
            sequence_length=160,
        )
        width = hf["input_ids"].shape[1]
        self.assertAllEqual(
            keras.ops.convert_to_numpy(x["token_ids"])[:, :width],
            hf["input_ids"].numpy(),
        )
        self.assertAllEqual(
            keras.ops.convert_to_numpy(x["padding_mask"])[:, :width],
            hf["attention_mask"].numpy().astype(bool),
        )
        self.assertAllEqual(
            keras.ops.convert_to_numpy(x["position_ids"])[:, :width],
            positions.numpy().transpose(1, 2, 0),
        )
        self.assert_pixels_match_hf(prompts, hf, x)
        return hf, x

    def assert_pixels_match_hf(self, prompts, hf, x):
        """Compare the real patches of each sample with HF's, item by item.

        HF stacks all images, then all videos, while KerasHub keeps each
        sample's media in prompt order.
        """
        p, t = 2, 2

        def hf_patches(key):
            if key not in hf:
                return None
            patches = hf[key].numpy().reshape(-1, 3, t, p, p)
            return patches.transpose(0, 2, 3, 4, 1)

        sources = {
            "image": hf_patches("pixel_values"),
            "video": hf_patches("pixel_values_videos"),
        }
        cursor = {"image": 0, "video": 0}
        pixels = keras.ops.convert_to_numpy(x["pixel_values"])
        grids = keras.ops.convert_to_numpy(x["image_grid_thw"])
        for b, prompt in enumerate(prompts):
            offset = 0
            kinds = re.findall(r"<\|(image|video)_pad\|>", prompt)
            for item, kind in enumerate(kinds):
                count = int(np.prod(grids[b, item]))
                start = cursor[kind]
                expected = sources[kind][start : start + count]
                actual = pixels[b, offset : offset + count]
                # Images without video keep one frame per patch.
                expected = expected[:, : actual.shape[1]]
                self.assertAllClose(actual, expected, atol=1e-6, rtol=0)
                cursor[kind] += count
                offset += count

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_image(self):
        self.assert_matches_hf([IMAGE_PROMPT], [[self.image]], [None], [None])

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_wrapped_video_marker(self):
        # The chat template wraps the marker in vision start and end tokens.
        self.assert_matches_hf([VIDEO_PROMPT], [None], [[self.video]], [[2.0]])

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_video_with_several_chunks(self):
        # 12 frames at 6 fps keep 4 frames, which is two temporal chunks.
        video = np.repeat(self.video, 2, axis=0)
        hf, x = self.assert_matches_hf(
            [BARE_VIDEO_PROMPT], [None], [[video]], [[6.0]]
        )
        self.assertAllEqual(hf["video_grid_thw"].numpy(), [[2, 4, 4]])
        self.assertAllEqual(x["image_grid_thw"], [[[2, 4, 4]]])

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_odd_frame_count(self):
        # 10 frames at 4 fps sample 5 frames at 2 fps. They are padded to 6,
        # and the last chunk repeats the last index in its timestamp.
        video = np.concatenate([self.video, self.video])[:10]
        indices = self.video_converter.sample_frame_indices(len(video), 4.0)
        self.assertLen(indices, 5)
        hf, x = self.assert_matches_hf(
            [BARE_VIDEO_PROMPT], [None], [[video]], [[4.0]]
        )
        self.assertAllEqual(hf["video_grid_thw"].numpy(), [[3, 4, 4]])

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_resized_images(self):
        rng = np.random.RandomState(1)
        for shape in (
            (8, 12, 3),  # Not square.
            # Fewer than `min_pixels`, so upscaled. A leading 3 would read as
            # channels first in HF.
            (2, 3, 3),
            (40, 40, 3),  # More than `max_pixels`, so downscaled.
        ):
            image = rng.randint(0, 255, shape).astype("uint8")
            self.assert_matches_hf([IMAGE_PROMPT], [[image]], [None], [None])

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_mixed_batch(self):
        video = np.repeat(self.video, 2, axis=0)
        prompts = [
            IMAGE_PROMPT + " " + BARE_VIDEO_PROMPT + " " + IMAGE_PROMPT,
            "air at",
            BARE_VIDEO_PROMPT,
        ]
        self.assert_matches_hf(
            prompts,
            [[self.image, self.image[::-1].copy()], None, []],
            [[video], None, [self.video]],
            [[6.0], None, [6.0]],
        )
