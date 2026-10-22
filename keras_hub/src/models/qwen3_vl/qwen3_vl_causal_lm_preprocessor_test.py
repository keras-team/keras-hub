import grain
import keras
import numpy as np
import pytest

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3_VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3_VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3_VLTokenizer
from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import (
    Qwen3_VLVideoConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3_VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.tokenizers.byte_pair_tokenizer import bytes_to_unicode

try:
    import tensorflow as tf
except ImportError:
    tf = None

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


class Qwen3_VLCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocabulary, self.merges = build_vocabulary()
        self.tokenizer = Qwen3_VLTokenizer(
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
        self.image_converter = Qwen3_VLImageConverter(**converter_kwargs)
        self.video_converter = Qwen3_VLVideoConverter(
            fps=2.0, min_frames=2, max_frames=8, **converter_kwargs
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "video_converter": self.video_converter,
            "sequence_length": 64,
        }
        self.preprocessor = Qwen3_VLCausalLMPreprocessor(**self.init_kwargs)
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
            cls=Qwen3_VLCausalLMPreprocessor,
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
        preprocessor = Qwen3_VLCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            preprocessor.generate_preprocess(
                {"prompts": [], "images": []}, sequence_length=16
            )

    def test_apply_chat_template(self):
        template = Qwen3_VLCausalLMPreprocessor.apply_chat_template
        messages = [
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "video"},
                    {"type": "text", "text": "What is shown?"},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ]
        vision = "<|vision_start|><|{}_pad|><|vision_end|>"
        expected = (
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            f"<|im_start|>user\n{vision.format('image')}"
            f"{vision.format('video')}What is shown?<|im_end|>\n"
            "<|im_start|>assistant\nA cat.<|im_end|>\n"
        )
        self.assertEqual(template(messages), expected)
        self.assertEqual(
            template(messages, add_generation_prompt=True),
            expected + "<|im_start|>assistant\n",
        )
        self.assertIn("Picture 1: ", template(messages, add_vision_id=True))
        self.assertIn("Video 1: ", template(messages, add_vision_id=True))
        with self.assertRaisesRegex(ValueError, "cannot hold"):
            template([{"role": "assistant", "content": [{"type": "image"}]}])
        with self.assertRaisesRegex(ValueError, "role"):
            template([{"role": "tool", "content": "x"}])
        with self.assertRaisesRegex(ValueError, "first message"):
            template([{"role": "user", "content": "a"}] + messages[:1])
        with self.assertRaisesRegex(ValueError, "empty"):
            template([])
        with self.assertRaisesRegex(ValueError, "content"):
            template([{"role": "assistant", "content": None}])
        self.assertEqual(
            template([{"role": "user", "content": ["a", {"type": "image"}]}]),
            template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "image"},
                        ],
                    }
                ]
            ),
        )
        # Hugging Face style media keys, and one counter across messages.
        prompt = template(
            [
                {"role": "user", "content": [{"image": "a.png"}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"image_url": "b"}, {"video": 1}]},
            ],
            add_vision_id=True,
        )
        self.assertIn("Picture 1: <|vision_start|><|image_pad|>", prompt)
        self.assertIn("Picture 2: <|vision_start|><|image_pad|>", prompt)
        self.assertIn("Video 1: <|vision_start|><|video_pad|>", prompt)

    def test_text_position_ids_skip_padding(self):
        x = self.preprocessor.generate_preprocess("air")
        length = len(self.tokenizer("air"))
        self.assertAllEqual(x["padding_mask"][:length], [True] * length)
        expected = np.zeros((64, 3), "int32")
        expected[:length] = np.arange(length)[:, None]
        self.assertAllEqual(x["position_ids"], expected)

    def test_start_and_end_tokens(self):
        preprocessor = Qwen3_VLCausalLMPreprocessor(
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

    def test_text_path_in_grain(self):
        length = len(self.tokenizer("air at"))
        ds = grain.MapDataset.source(["air at", "air"])
        for x in ds.map(self.preprocessor.generate_preprocess):
            assert_grain_safe_types(x)
            self.assertEqual(x["position_ids"].shape, (64, 3))
        ds = grain.MapDataset.source([{"prompts": "air at"}])
        x, y, sw = next(iter(ds.map(self.preprocessor)))
        assert_grain_safe_types((x, y, sw))
        self.assertAllEqual(x["position_ids"][:length, 0], range(length))
        self.assertEqual(y.shape, (64,))

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
        self.assertEqual(x["pixel_values"].shape, (16, 2, 2, 2, 3))
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
        mask = keras.ops.convert_to_numpy(x["padding_mask"]).reshape(-1)
        self.assertFalse(mask[indices[1]].any())
        self.assertTrue(((indices[1] // 64) == 1).all())

    def test_image_features_scatter_into_vision_tokens(self):
        x = self.preprocessor.generate_preprocess(
            {
                "prompts": [IMAGE_PROMPT, IMAGE_PROMPT],
                "images": [[self.image]] * 2,
            }
        )
        token_ids = keras.ops.convert_to_numpy(x["token_ids"]).reshape(-1)
        indices = keras.ops.convert_to_numpy(x["vision_indices"]).reshape(-1)
        self.assertTrue((token_ids[indices] == self.tok("<|image_pad|>")).all())
        self.assertEqual(len(set(indices.tolist())), 8)

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
        self.assertEqual(x["pixel_values"].shape, (16, 2, 2, 2, 3))
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
        self.assertEqual(
            int(keras.ops.convert_to_numpy(sw)[0].sum()), length - 1
        )

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

    def test_truncated_vision_tokens_raise(self):
        with self.assertRaisesRegex(ValueError, "do not fit"):
            self.preprocessor.generate_preprocess(
                {"prompts": IMAGE_PROMPT, "images": [self.image]},
                sequence_length=5,
            )

    def test_config_round_trip(self):
        preprocessor = Qwen3_VLCausalLMPreprocessor(
            **self.init_kwargs, default_video_fps=30.0, add_end_token=False
        )
        config = preprocessor.get_config()
        restored = Qwen3_VLCausalLMPreprocessor.from_config(dict(config))
        self.assertEqual(restored.get_config(), config)
        self.assertEqual(restored.default_video_fps, 30.0)
        self.assertFalse(restored.add_end_token)
        self.assertIsInstance(restored.image_converter, Qwen3_VLImageConverter)
        self.assertIsInstance(restored.video_converter, Qwen3_VLVideoConverter)
        self.assertEqual(restored.video_converter.patch_size, 2)

    def test_default_video_fps_is_used(self):
        preprocessor = Qwen3_VLCausalLMPreprocessor(
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
        vision_encoder = Qwen3_VLVisionEncoder(
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
        backbone = Qwen3_VLBackbone(
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
        return hf, x

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
        # 5 sampled frames are padded to 6, and the last chunk repeats the
        # last index in its timestamp.
        video = np.concatenate([self.video, self.video])[:10]
        self.assert_matches_hf([BARE_VIDEO_PROMPT], [None], [[video]], [[2.0]])

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
