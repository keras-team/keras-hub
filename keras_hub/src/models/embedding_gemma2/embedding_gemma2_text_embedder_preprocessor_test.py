import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_audio_converter import (  # noqa: E501
    EmbeddingGemma2AudioConverter,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_image_converter import (  # noqa: E501
    EmbeddingGemma2ImageConverter,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder_preprocessor import (  # noqa: E501
    EmbeddingGemma2TextEmbedderPreprocessor,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_tokenizer import (
    EmbeddingGemma2Tokenizer,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_video_converter import (  # noqa: E501
    EmbeddingGemma2VideoConverter,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2TextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer(add_bos=False, add_eos=False)

        self.image_converter = EmbeddingGemma2ImageConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_soft_tokens=1,
        )
        self.audio_converter = EmbeddingGemma2AudioConverter(
            num_mels=8,
            num_fft_bins=16,
            stride=8,
            max_audio_length=1,
            sampling_rate=160,
        )
        self.video_converter = EmbeddingGemma2VideoConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=2,
            max_soft_tokens=1,
        )

        self.preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            audio_converter=self.audio_converter,
            video_converter=self.video_converter,
            sequence_length=12,
        )

    def _both_paths(self, **kwargs):
        """The python-path preprocessor and its TF-path twin.

        `_allow_python_workflow=False` forces `_call_tf` in eager mode; it is
        the switch a user has, unlike patching `_use_tf_workflow`.
        """
        init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "audio_converter": self.audio_converter,
            "video_converter": self.video_converter,
            "sequence_length": 12,
            **kwargs,
        }
        return (
            EmbeddingGemma2TextEmbedderPreprocessor(**init_kwargs),
            EmbeddingGemma2TextEmbedderPreprocessor(
                _allow_python_workflow=False, **init_kwargs
            ),
        )

    def test_preprocessor_text_only(self):
        out = self.preprocessor(["the quick brown fox"])
        if isinstance(out, tuple):
            out = out[0]
        # the=9, quick=14, brown=10, fox=12
        # bos=1, eos=2, pad=0
        expected = [1, 9, 14, 10, 12, 2, 0, 0, 0, 0, 0, 0]
        self.assertAllEqual(ops.convert_to_numpy(out["token_ids"])[0], expected)
        self.assertAllEqual(
            ops.convert_to_numpy(out["padding_mask"])[0],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],  # noqa: E501
        )

    def test_preprocessor_text_only_with_multimodal_backbone(self):
        out = self.preprocessor(["the quick brown fox"])
        if isinstance(out, tuple):
            out = out[0]
        # Verify placeholders are emitted
        self.assertIn("pixel_values", out)
        self.assertIn("audio_indices", out)
        self.assertEqual(ops.shape(out["pixel_values"])[1], 0)
        self.assertEqual(ops.shape(out["audio_mel"])[1], 0)

    def test_preprocessor_ragged_images(self):
        image_converter = EmbeddingGemma2ImageConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_soft_tokens=4,
        )
        preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=32,
        )
        img1 = np.ones((16, 16, 3), dtype="float32")
        img2 = np.ones((32, 16, 3), dtype="float32")
        images = [img1, img2]
        out = preprocessor({"images": images})
        if isinstance(out, tuple):
            out = out[0]
        self.assertIn("pixel_values", out)

        # `vision_indices` is right-padded with 0; position 0 is always BOS,
        # so non-zero entries are exactly the soft-token slots.
        indices = ops.convert_to_numpy(out["vision_indices"])
        count1 = np.count_nonzero(indices[0])
        count2 = np.count_nonzero(indices[1])
        # img1 is 16x16: square fills the 4-token budget (2x2 patches).
        # img2 is 32x16: 2:1 aspect snaps to a 2x1 patch grid -> 2 tokens.
        self.assertEqual(count1, 4)
        self.assertEqual(count2, 2)

    def test_preprocessor_ragged_videos(self):
        # Custom preprocessor: max_frames=32 and a sequence_length long
        # enough to hold 32 frames of tokens.
        video_converter = EmbeddingGemma2VideoConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=32,
            max_soft_tokens=1,
        )
        preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=100,
        )
        vid1 = np.ones((5, 16, 16, 3), dtype="float32")
        vid2 = np.ones((40, 16, 16, 3), dtype="float32")
        videos = [vid1, vid2]
        out = preprocessor({"videos": videos})
        if isinstance(out, tuple):
            out = out[0]
        self.assertIn("pixel_values", out)

        indices = ops.convert_to_numpy(out["vision_indices"])
        count1 = np.count_nonzero(indices[0])
        count2 = np.count_nonzero(indices[1])
        # 5 frames -> 5 tokens, 40 frames -> 32 tokens
        self.assertEqual(count1, 5)
        self.assertEqual(count2, 32)

    def test_preprocessor_ragged_audio(self):
        # 8000 samples -> 49 frames -> 13 tokens
        # 20800 samples -> 129 frames -> 33 tokens
        audio_converter = EmbeddingGemma2AudioConverter(
            num_mels=8,
            num_fft_bins=16,
            stride=160,
            max_audio_length=150,
            sampling_rate=160,
        )
        preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=audio_converter,
            sequence_length=100,
        )

        aud1 = np.ones((8000,), dtype="float32")
        aud2 = np.ones((20800,), dtype="float32")
        audio = [aud1, aud2]
        out = preprocessor({"audio": audio})
        if isinstance(out, tuple):
            out = out[0]
        self.assertIn("audio_mel", out)

        indices = ops.convert_to_numpy(out["audio_indices"])
        count1 = np.count_nonzero(indices[0])
        count2 = np.count_nonzero(indices[1])
        self.assertEqual(count1, 13)
        self.assertEqual(count2, 33)

        mel_mask = ops.convert_to_numpy(out["audio_mel_mask"])
        mel_count1 = np.sum(mel_mask[0])
        mel_count2 = np.sum(mel_mask[1])
        self.assertEqual(mel_count1, 49)
        self.assertEqual(mel_count2, 129)

    def test_preprocessor_image_expansion(self):
        images = ops.ones((1, 16, 16, 3))
        out = self.preprocessor({"images": images})
        if isinstance(out, tuple):
            out = out[0]

        expected = [1, 4, 8, 5, 2, 0, 0, 0, 0, 0, 0, 0]
        self.assertAllEqual(ops.convert_to_numpy(out["token_ids"])[0], expected)
        self.assertIn("pixel_values", out)
        self.assertIn("vision_indices", out)

    def test_preprocessor_audio_expansion(self):
        audio = ops.ones((1, 160))
        out = self.preprocessor({"audio": audio})
        if isinstance(out, tuple):
            out = out[0]

        expected = [1, 20, 19, 19, 19, 19, 19, 21, 2, 0, 0, 0]
        self.assertAllEqual(ops.convert_to_numpy(out["token_ids"])[0], expected)
        self.assertIn("audio_mel", out)
        self.assertIn("audio_indices", out)

    def test_preprocessor_video_expansion(self):
        videos = ops.ones((1, 2, 16, 16, 3))  # 2 frames
        out = self.preprocessor({"videos": videos})
        if isinstance(out, tuple):
            out = out[0]

        expected = [1, 4, 22, 5, 4, 22, 5, 2, 0, 0, 0, 0]
        self.assertAllEqual(ops.convert_to_numpy(out["token_ids"])[0], expected)
        self.assertIn("pixel_values", out)
        self.assertIn("vision_indices", out)

    def test_run_preprocessing_layer_test(self):
        self.run_preprocessing_layer_test(
            cls=EmbeddingGemma2TextEmbedderPreprocessor,
            init_kwargs={
                "tokenizer": self.tokenizer,
                "sequence_length": 8,
            },
            input_data=["the quick brown fox"],
            expected_output={
                "token_ids": [[1, 9, 14, 10, 12, 2, 0, 0]],
                "padding_mask": [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                    ]
                ],
            },
            expected_detokenize_output=None,
        )

    def test_serialization(self):
        self.run_serialization_test(self.preprocessor)

    def test_call_with_python_only_tokenizer(self):
        tokenizer = EmbeddingGemma2Tokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma4_test_vocab.spm"
            )
        )
        image_converter = EmbeddingGemma2ImageConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_soft_tokens=1,
        )
        preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=tokenizer,
            image_converter=image_converter,
            sequence_length=12,
        )

        texts = ["the quick brown fox", "short"]

        # Eager calls take the python path, so this pins `_call_python`.
        out = preprocessor({"texts": texts})

        token_ids = ops.convert_to_numpy(out["token_ids"])
        self.assertEqual(token_ids.shape, (2, 12))

        tok_out = tokenizer(texts)

        exp_1 = (
            [tokenizer.start_token_id] + tok_out[0] + [tokenizer.end_token_id]
        )
        exp_1 = exp_1 + [tokenizer.pad_token_id] * (12 - len(exp_1))

        exp_2 = (
            [tokenizer.start_token_id] + tok_out[1] + [tokenizer.end_token_id]
        )
        exp_2 = exp_2 + [tokenizer.pad_token_id] * (12 - len(exp_2))

        self.assertAllEqual(token_ids[0], exp_1)
        self.assertAllEqual(token_ids[1], exp_2)

    def test_call_with_tf_ragged_media(self):
        # Eager `tf.RaggedTensor` media must be accepted by the python path
        # and give the same result as plain python lists. One modality per
        # call: Keras rejects a nested argument that mixes a python list
        # (`texts`) with a tensor on the TensorFlow backend.
        tf = pytest.importorskip("tensorflow")
        rng = np.random.RandomState(0)
        audio = [
            rng.uniform(-1, 1, (4000,)).astype("float32"),
            rng.uniform(-1, 1, (6000,)).astype("float32"),
        ]
        videos = [
            [rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)],
            [
                rng.randint(0, 256, (16, 16, 3), dtype=np.uint8),
                rng.randint(0, 256, (16, 16, 3), dtype=np.uint8),
            ],
        ]
        for key, plain in (("audio", audio), ("videos", videos)):
            plain_out = self.preprocessor({key: plain})
            ragged_out = self.preprocessor({key: tf.ragged.constant(plain)})
            for k in plain_out:
                self.assertAllEqual(
                    ops.convert_to_numpy(plain_out[k]),
                    ops.convert_to_numpy(ragged_out[k]),
                )

    @pytest.mark.tf_only
    def test_tf_data_map_matches_eager(self):
        # The TF path runs inside `tf.data.Dataset.map`, where every tensor
        # is symbolic and the batch dim is unknown. Each modality must trace
        # there and produce exactly the eager result.
        import tensorflow as tf

        rng = np.random.RandomState(0)
        cases = {
            "images": {
                "texts": ["<|image|> a", "<|image|> b"],
                "images": rng.uniform(0, 255, (2, 16, 16, 3)).astype("float32"),
            },
            "videos": {
                "texts": ["<|video|> a", "<|video|> b"],
                "videos": rng.uniform(0, 255, (2, 2, 16, 16, 3)).astype(
                    "float32"
                ),
            },
            "audio": {
                "texts": ["<|audio|> a", "<|audio|> b"],
                "audio": rng.uniform(-1, 1, (2, 320)).astype("float32"),
            },
        }
        for name, inputs in cases.items():
            eager = self.preprocessor(inputs)
            if isinstance(eager, tuple):
                eager = eager[0]
            ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)
            mapped = next(iter(ds.map(self.preprocessor)))
            if isinstance(mapped, tuple):
                mapped = mapped[0]
            self.assertEqual(set(mapped.keys()), set(eager.keys()), name)
            for key in eager:
                got = ops.convert_to_numpy(mapped[key])
                want = ops.convert_to_numpy(eager[key])
                if want.dtype.kind == "f":
                    # TF picks a different STFT kernel under tracing; the
                    # mel differs by ~1e-6 on values of order 10.
                    self.assertAllClose(
                        got, want, atol=1e-5, msg=f"{name}: {key}"
                    )
                else:
                    self.assertAllEqual(got, want, msg=f"{name}: {key}")

    @pytest.mark.tf_only
    def test_tf_data_map_runs_validation(self):
        # The placeholder and truncation checks are `tf.debugging` asserts
        # on the TF path; they must not be pruned when the preprocessor is
        # traced inside `tf.data.Dataset.map`.
        import tensorflow as tf

        img = np.ones((2, 16, 16, 3), dtype="float32")
        cases = {
            "two placeholders": (
                {
                    "texts": ["<|image|> <|image|>", "<|image|> b"],
                    "images": img,
                },
                "exactly 1 image placeholder",
            ),
            "placeholder, no media": (
                {"texts": ["<|image|> a", "b"]},
                "image placeholder but no media",
            ),
            "truncation": (
                {
                    "texts": ["a b c d e f g h i j <|image|>", "<|image|> b"],
                    "images": img,
                },
                "soft tokens do not fit",
            ),
        }
        for name, (inputs, message) in cases.items():
            ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)
            with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, message, msg=name
            ):
                next(iter(ds.map(self.preprocessor)))

    def test_images_and_videos_together_raise(self):
        img = np.ones((16, 16, 3), dtype="float32")
        vid = np.ones((2, 16, 16, 3), dtype="float32")
        inputs = {
            "texts": ["<|image|> and <|video|>"],
            "images": [img],
            "videos": [vid],
        }
        # Both execution paths must reject the combination.
        for preprocessor in self._both_paths():
            with self.assertRaisesRegex(
                ValueError, "`images` and `videos` in the same call"
            ):
                preprocessor(inputs)

    def test_multiple_images_per_prompt_raise(self):
        img = np.ones((16, 16, 3), dtype="float32")
        two_images = np.ones((1, 2, 16, 16, 3), dtype="float32")
        # The check runs before any converter call, so it can be exercised
        # on both paths regardless of backend.
        for preprocessor in self._both_paths():
            with self.assertRaisesRegex(ValueError, "one image per prompt"):
                preprocessor({"images": two_images})
            with self.assertRaisesRegex(ValueError, "one image per prompt"):
                preprocessor({"images": [[img, img]]})
        # One image per prompt is still fine in both spellings.
        self.preprocessor({"images": img})
        self.preprocessor({"images": [img, img]})

    # --- Round-4 review: input validation, truncation, prompts, layout ---

    def _token_count(self, out, placeholder):
        ids = ops.convert_to_numpy(out["token_ids"])
        return int((ids == self.tokenizer.token_to_id(placeholder)).sum())

    def test_media_token_truncation_raises(self):
        # 2 frames x 1 soft token + wrappers + 8 words do not fit in 12.
        vid = np.ones((2, 16, 16, 3), dtype="float32")
        long_text = "a b c d e f g h <|video|>"
        for preprocessor in self._both_paths():
            with self.assertRaisesRegex(
                Exception,
                "video soft tokens do not fit in `sequence_length=12`",
            ):
                preprocessor({"texts": [long_text], "videos": [vid]})
            # The same input fits once the budget is raised.
            out = preprocessor(
                {"texts": [long_text], "videos": [vid]}, sequence_length=32
            )
            self.assertEqual(self._token_count(out, "<|video|>"), 2)

    def test_placeholder_media_mismatch_raises(self):
        img = np.ones((16, 16, 3), dtype="float32")
        aud = np.ones((400,), dtype="float32")
        cases = [
            # Two placeholders, one image.
            (
                {"texts": ["<|image|> and <|image|>"], "images": [img]},
                "exactly 1 image",
            ),
            # Image with no placeholder.
            ({"texts": ["hello"], "images": [img]}, "exactly 1 image"),
            # Placeholder with no media.
            ({"texts": ["<|image|> hello"]}, "image placeholder but no media"),
            ({"texts": ["<|audio|> hello"]}, "audio placeholder but no media"),
            # Two texts, one image.
            (
                {"texts": ["<|image|> a", "<|image|> b"], "images": [img]},
                "Batch size of texts and image",
            ),
            # Two audio clips, one text.
            (
                {"texts": ["<|audio|>"], "audio": [aud, aud]},
                "Batch size of texts and audio",
            ),
        ]
        for preprocessor in self._both_paths():
            for inputs, message in cases:
                with self.assertRaisesRegex(Exception, message):
                    preprocessor(inputs)

    def test_single_item_media_with_list_text(self):
        # The docstring's single-item forms: a list with one text plus a bare
        # rank-3 image or rank-1 waveform is one example, on both paths.
        img = np.ones((16, 16, 3), dtype="float32")
        aud = np.ones((400,), dtype="float32")
        for preprocessor in self._both_paths():
            bare = preprocessor({"texts": ["<|image|> a"], "images": img})
            listed = preprocessor({"texts": ["<|image|> a"], "images": [img]})
            for key in listed:
                self.assertAllEqual(
                    ops.convert_to_numpy(bare[key]),
                    ops.convert_to_numpy(listed[key]),
                )
            bare = preprocessor({"texts": ["<|audio|> a"], "audio": aud})
            listed = preprocessor({"texts": ["<|audio|> a"], "audio": [aud]})
            for key in listed:
                self.assertAllEqual(
                    ops.convert_to_numpy(bare[key]),
                    ops.convert_to_numpy(listed[key]),
                )

    def test_texts_none_builds_prompt_from_every_modality(self):
        img = np.ones((16, 16, 3), dtype="float32")
        aud = np.ones((400,), dtype="float32")
        for preprocessor in self._both_paths():
            out = preprocessor({"images": [img], "audio": [aud]})
            # One image soft token and the audio soft tokens, as with the
            # explicit prompt "<|image|> <|audio|>".
            explicit = preprocessor(
                {
                    "texts": ["<|image|> <|audio|>"],
                    "images": [img],
                    "audio": [aud],
                }
            )
            self.assertAllEqual(
                ops.convert_to_numpy(out["token_ids"]),
                ops.convert_to_numpy(explicit["token_ids"]),
            )
            self.assertEqual(self._token_count(out, "<|image|>"), 1)
            self.assertGreater(self._token_count(out, "<|audio|>"), 0)
            # Inconsistent batch sizes across modalities are rejected.
            with self.assertRaisesRegex(Exception, "inconsistently sized"):
                preprocessor({"images": [img, img], "audio": [aud]})

    def test_long_audio_keeps_hf_frame_count(self):
        # A clip longer than the converter window keeps
        # `(num_samples - 1) // stride` frames, like HF, not one more.
        converter = self.audio_converter
        clip = np.ones((converter.num_samples + 40,), dtype="float32")
        expected_frames = (converter.num_samples - 1) // converter.stride
        for preprocessor in self._both_paths():
            out = preprocessor({"audio": [clip]})
            mask = ops.convert_to_numpy(out["audio_mel_mask"])
            self.assertEqual(int(mask.sum()), expected_frames)
            self.assertEqual(
                ops.convert_to_numpy(out["audio_mel"]).shape[2],
                expected_frames,
            )

    def test_video_indices_are_frame_major_per_row(self):
        # Clips whose frames give different soft-token counts: every row is
        # laid out frame by frame with the same width, so the narrower clip
        # never receives another clip's padding slots (keras-hub#3144 T9).
        video_converter = EmbeddingGemma2VideoConverter(
            image_size=(16, 16),
            patch_size=8,
            pooling_kernel_size=1,
            max_frames=2,
            max_soft_tokens=4,
        )
        wide = np.ones((2, 16, 16, 3), dtype="float32")  # 4 tokens / frame
        narrow = np.ones((2, 32, 16, 3), dtype="float32")  # fewer / frame
        for preprocessor in self._both_paths(
            video_converter=video_converter, sequence_length=32
        ):
            batch = preprocessor({"videos": [wide, narrow]})
            alone = preprocessor({"videos": [narrow]})
            batch_idx = ops.convert_to_numpy(batch["vision_indices"])
            alone_idx = ops.convert_to_numpy(alone["vision_indices"])
            tokens_per_frame = {
                "wide": self._token_count(
                    preprocessor({"videos": [wide]}), "<|video|>"
                )
                // 2,
                "narrow": self._token_count(alone, "<|video|>") // 2,
            }
            self.assertGreater(
                tokens_per_frame["wide"], tokens_per_frame["narrow"]
            )
            width = tokens_per_frame["wide"]
            self.assertEqual(batch_idx.shape, (2, 2 * width))
            # Row 1 (narrow): each frame block starts with its real
            # positions and is zero-padded to `width`.
            n = tokens_per_frame["narrow"]
            for f in range(2):
                block = batch_idx[1, f * width : (f + 1) * width]
                self.assertAllEqual(
                    block[:n], alone_idx[0, f * n : (f + 1) * n]
                )
                self.assertAllEqual(block[n:], np.zeros(width - n))

    def test_preprocessor_batched_vs_per_image(self):
        import unittest.mock

        from keras_hub.src.models.embedding_gemma2 import (
            embedding_gemma2_image_converter,
        )
        EmbeddingGemma2ImageConverter = (
            embedding_gemma2_image_converter.EmbeddingGemma2ImageConverter
        )

        original_call = EmbeddingGemma2ImageConverter.call
        call_count = [0]

        def mocked_call(self_obj, *args, **kwargs):
            call_count[0] += 1
            return original_call(self_obj, *args, **kwargs)

        with unittest.mock.patch.object(
            EmbeddingGemma2ImageConverter,
            "call",
            autospec=True,
            side_effect=mocked_call,
        ):
            img1 = np.ones((16, 16, 3))
            img2 = np.ones((16, 16, 3))

            # Uniform shape: should be batched
            call_count[0] = 0
            out_batch = self.preprocessor({"images": [img1, img2]})
            self.assertEqual(call_count[0], 1)

            # Per-image baseline
            call_count[0] = 0
            out_img1 = self.preprocessor({"images": [img1]})
            out_img2 = self.preprocessor({"images": [img2]})
            self.assertEqual(call_count[0], 2)

            # Assert batched outputs equal per-image outputs
            self.assertAllClose(
                out_batch["pixel_values"][0], out_img1["pixel_values"][0]
            )
            self.assertAllClose(
                out_batch["pixel_values"][1], out_img2["pixel_values"][0]
            )

            # Mixed shape: should fall back to per-image calls
            img3 = np.ones((12, 12, 3))
            call_count[0] = 0
            self.preprocessor({"images": [img1, img3]})
            self.assertEqual(call_count[0], 2)

    def test_preprocessor_shares_modality_expansion(self):
        # T5: Test the extracted helper directly. If the old code lacked this
        # helper, this test would fail because the attribute does not exist.
        texts = ["A <|image|> B", "C <|audio|> D"]
        tokens_list = [2, 3]

        img_out = self.preprocessor._expand_modality_placeholders_py(
            texts.copy(),
            tokens_list,
            self.preprocessor.image_placeholder,
            self.preprocessor.start_of_image_token,
            self.preprocessor.end_of_image_token,
        )
        expected_img = (
            "A "
            + self.preprocessor.start_of_image_token
            + self.preprocessor.image_placeholder * 2
            + self.preprocessor.end_of_image_token
            + " B"
        )
        self.assertEqual(img_out[0], expected_img)

        aud_out = self.preprocessor._expand_modality_placeholders_py(
            texts.copy(),
            tokens_list,
            self.preprocessor.audio_placeholder,
            self.preprocessor.start_of_audio_token,
            self.preprocessor.end_of_audio_token,
        )
        expected_aud = (
            "C "
            + self.preprocessor.start_of_audio_token
            + self.preprocessor.audio_placeholder * 3
            + self.preprocessor.end_of_audio_token
            + " D"
        )
        self.assertEqual(aud_out[1], expected_aud)
