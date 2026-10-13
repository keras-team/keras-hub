import os
from unittest import mock

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

        # Force the python path so this pins `_call_python`, not the TF path.
        with mock.patch.object(
            preprocessor, "_use_tf_workflow", return_value=False
        ):
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
            with mock.patch.object(
                self.preprocessor, "_use_tf_workflow", return_value=False
            ):
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

    def test_images_and_videos_together_raise(self):
        img = np.ones((16, 16, 3), dtype="float32")
        vid = np.ones((2, 16, 16, 3), dtype="float32")
        inputs = {
            "texts": ["<|image|> and <|video|>"],
            "images": [img],
            "videos": [vid],
        }
        # Both execution paths must reject the combination.
        for use_tf in (False, True):
            if use_tf and pytest.importorskip("tensorflow") is None:
                continue
            with mock.patch.object(
                self.preprocessor, "_use_tf_workflow", return_value=use_tf
            ):
                with self.assertRaisesRegex(
                    ValueError, "`images` and `videos` in the same call"
                ):
                    self.preprocessor(inputs)

    def test_multiple_images_per_prompt_raise(self):
        img = np.ones((16, 16, 3), dtype="float32")
        two_images = np.ones((1, 2, 16, 16, 3), dtype="float32")
        # The check runs before any converter call, so it can be exercised
        # on both paths regardless of backend.
        for use_tf in (False, True):
            with mock.patch.object(
                self.preprocessor, "_use_tf_workflow", return_value=use_tf
            ):
                with self.assertRaisesRegex(ValueError, "one image per prompt"):
                    self.preprocessor({"images": two_images})
                with self.assertRaisesRegex(ValueError, "one image per prompt"):
                    self.preprocessor({"images": [[img, img]]})
        # One image per prompt is still fine in both spellings.
        self.preprocessor({"images": img})
        self.preprocessor({"images": [img, img]})
