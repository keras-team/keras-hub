import keras
import numpy as np
import pytest
import tensorflow as tf

from keras_hub.src.models.gemma4_unified.gemma4_unified_causal_lm_preprocessor import (  # noqa: E501
    Gemma4UnifiedCausalLMPreprocessor,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_image_converter import (
    Gemma4UnifiedImageConverter,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_video_converter import (
    Gemma4UnifiedVideoConverter,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.mocks.mock_gemma4_unified_fixtures import (
    raw_image_preprocessor_kwargs,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedCausalLMPreprocessorTest(TestCase):
    def test_image_truncation_raises(self):
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=8,
            num_vision_tokens_per_image=5,
            max_images_per_prompt=1,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Image placeholders were truncated. "
            "Please raise `sequence_length`.",
        ):
            preprocessor.generate_preprocess(
                {
                    "prompts": ["a b c d e f g h i j k l <|image|> text"],
                    "images": [tf.zeros([1, 16, 16, 3])],
                }
            )

    def test_tf_data_map(self):
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=300,
            num_vision_tokens_per_image=10,
        )
        # Test 1: preprocessor (training mode) with prompts, responses, images
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "prompts": ["<|image|> text", "text <|image|>"],
                "responses": ["response 1", "response 2"],
                "images": [tf.zeros([1, 16, 16, 3]), tf.zeros([1, 16, 16, 3])],
            }
        ).batch(2)
        ds = ds.map(preprocessor)
        for batch in ds:
            self.assertIn("token_ids", batch[0])
            self.assertIsNotNone(batch[1])  # labels

        # Test 2: generate_preprocess with prompts and images only
        ds_gen = tf.data.Dataset.from_tensor_slices(
            {
                "prompts": ["<|image|> text", "text <|image|>"],
                "images": [tf.zeros([1, 16, 16, 3]), tf.zeros([1, 16, 16, 3])],
            }
        ).batch(2)
        ds_gen = ds_gen.map(preprocessor.generate_preprocess)
        for batch in ds_gen:
            self.assertIn("token_ids", batch)

    def test_per_sample_image_counts(self):
        # We need a converter that returns different valid masks.
        # But our dummy converter returns fixed sizes.
        # Let's mock the converter to return different pixel_position_ids.
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=300,
            num_vision_tokens_per_image=50,
        )
        # Mock pixel_position_ids for 2 samples: 1 with 20 tokens, 1 with 30
        # tokens. We'll just test that the placeholder replacement in
        # generate_preprocess works!
        pixel_position_ids = tf.constant(
            [
                [[[1, 1]] * 20 + [[-1, -1]] * 30],
                [[[1, 1]] * 30 + [[-1, -1]] * 20],
            ]
        )

        prompts = tf.constant(["<|image|> A", "<|image|> B"])
        expanded, expected_vision = preprocessor._expand_image_prompt(
            prompts, pixel_position_ids
        )
        expanded_str = [p.numpy().decode("utf-8") for p in expanded]

        self.assertEqual(
            expanded_str[0].count(preprocessor.image_placeholder), 20
        )
        self.assertEqual(
            expanded_str[1].count(preprocessor.image_placeholder), 30
        )
        # text after image survives
        self.assertTrue(expanded_str[0].endswith(" A"))
        self.assertTrue(expanded_str[1].endswith(" B"))

    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # === Text Preprocessor ===
        self.init_text_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "sequence_length": 8,
            "max_images_per_prompt": 0,
            "num_vision_tokens_per_image": 0,
        }
        self.text_preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            sequence_length=100,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
        )

        # === Text + Image Preprocessor ===
        self.image_converter = Gemma4UnifiedImageConverter(
            image_size=(4, 4),
            patch_size=4,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 20,
            "max_images_per_prompt": 2,
            "num_vision_tokens_per_image": 5,
        }

    def test_text_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            **self.init_text_kwargs
        )
        self.run_serialization_test(preprocessor)

        output = preprocessor(input_data)

        expected_output_batched = (
            {
                "token_ids": [[1, 9, 14, 10, 12, 15, 2, 0]],
                "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
            },
            [[9, 14, 10, 12, 15, 2, 0, 0]],  # Labels shifted.
            [[0, 0, 0, 0, 1, 1, 0, 0]],  # Zero out unlabeled examples.
        )

        self.assertAllClose(output, expected_output_batched)

    def test_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox <|image|>"],
            "responses": ["round"],
            "pixel_values": np.ones((1, 2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((1, 2, 1, 2), dtype="int32"),
        }
        output = self.run_preprocessing_layer_test(
            cls=Gemma4UnifiedCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            return_output=True,
        )

        expected_output = [
            {
                "vision_indices": [[6] + [0] * 9],
                "vision_mask": [[0] * 6 + [1] * 1 + [0] * 13],
                "token_ids": [
                    [1, 9, 14, 10, 12, 4] + [8] * 1 + [5, 15, 2] + [0] * 10
                ],
                "padding_mask": [[1] * 10 + [0] * 10],
            },
            [
                [9, 14, 10, 12, 4] + [8] * 1 + [5, 15, 2] + [0] * 11
            ],  # Labels shifted.
            [[0] * 7 + [1] * 2 + [0] * 11],  # Zero out unlabeled examples.
        ]

        # Check pixel_values shape.
        self.assertAllEqual(output[0]["pixel_values"].shape, [1, 2, 1, 48])

        # Check the remaining values.
        del output[0]["pixel_values"]
        del output[0]["pixel_position_ids"]
        for key in expected_output[0].keys():
            self.assertAllEqual(output[0][key], expected_output[0][key])
        self.assertAllEqual(output[1], expected_output[1])
        self.assertAllEqual(output[2], expected_output[2])

    def test_text_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 4,
            "responses": ["round"] * 4,
        }
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            **self.init_text_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[9, 14, 10, 12, 15, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[14, 10, 12, 15, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[0, 0, 0, 1, 0, 0, 0, 0]] * 4)

    def test_text_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            **self.init_text_kwargs
        )
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 9, 14, 10, 12, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess(self):
        input_data = {
            "prompts": "the quick brown fox <|image|>",
            "pixel_values": np.ones((2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((2, 1, 2), dtype="int32"),
        }
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [1, 9, 14, 10, 12, 4] + [8] * 1 + [5] + [0] * 12,
        )
        self.assertAllEqual(x["padding_mask"], [1] * 8 + [0] * 12)
        self.assertAllEqual(x["vision_indices"], [6] + [0] * 9)
        self.assertAllEqual(x["vision_mask"], [0] * 6 + [1] * 1 + [0] * 13)
        self.assertAllEqual(x["pixel_values"].shape, [2, 1, 48])

    def test_text_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            **self.init_text_kwargs
        )
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    def test_text_only_predict(self):
        # F4 Regression: No responses key should not raise KeyError
        x = {"prompts": ["hello world"]}
        out = self.text_preprocessor(x)
        self.assertIn("token_ids", out)

    def test_backend_tensor_images(self):
        # F5 regression: a list of backend tensors next to string prompts
        # must not trip Keras' "cannot mix tensors and non-tensors" guard
        # in `Layer.__call__`, and must reach `generate_preprocess` even
        # though its decorator calls `np.array` on lists first. On torch
        # with CUDA the tensors live on the GPU; on TensorFlow an eager
        # `tf.Tensor` is the backend tensor.
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer, **raw_image_preprocessor_kwargs()
        )
        prompts = ["<|image|> a", "<|image|> b"]
        images = [keras.ops.zeros((16, 16, 3)), keras.ops.zeros((16, 16, 3))]
        x = {"prompts": prompts, "images": images}
        out = preprocessor(x)
        self.assertEqual(out["pixel_values"].shape[0], 2)
        out = preprocessor.generate_preprocess(x)
        self.assertIn("vision_mask", out)
        if keras.config.backend() == "tensorflow":
            # Every leaf eager, the form `next(iter(dataset))` yields: string
            # tensors must be coerced together with the images or the dict
            # becomes a tensor / non-tensor mix.
            x = {
                "prompts": tf.constant(prompts),
                "images": tf.zeros([2, 16, 16, 3]),
            }
            out = preprocessor(x)
            self.assertEqual(out["pixel_values"].shape[0], 2)

    def test_eager_scalar_string_tensors(self):
        # TF backend: eager string tensors are handed over as Python values,
        # so a rank-0 prompt still takes the unbatched path in `call` and
        # comes back with rank-1 token ids. Keeping the 0-d object array
        # instead raises inside `call`.
        if keras.config.backend() != "tensorflow":
            self.skipTest("Eager `tf.Tensor`s are backend tensors on TF only.")
        x = {
            "prompts": tf.constant("hello world"),
            "responses": tf.constant("a b"),
        }
        out = self.text_preprocessor(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        self.assertEqual(
            tuple(out[0]["token_ids"].shape),
            (self.text_preprocessor.sequence_length,),
        )

    def test_explicit_labels_pass_through_without_responses(self):
        # Without `responses` an explicitly supplied `(x, y, sample_weight)`
        # is returned unchanged rather than silently dropped.
        y_in = np.array([[7, 8, 9, 10]], dtype="int32")
        sw_in = np.array([[0, 1, 1, 0]], dtype="int32")
        x_in = {"prompts": ["hello world"]}
        out = self.text_preprocessor((x_in, y_in, sw_in))
        self.assertIsInstance(out, tuple)
        x, y, sw = out
        self.assertIn("token_ids", x)
        self.assertAllEqual(y, y_in)
        self.assertAllEqual(sw, sw_in)

        preprocessor = Gemma4UnifiedCausalLMPreprocessor(**self.init_kwargs)
        x_in = {
            "prompts": ["the quick brown fox <|image|>"],
            "pixel_values": np.ones((1, 2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((1, 2, 1, 2), dtype="int32"),
        }
        self.assertIsInstance(preprocessor(x_in), dict)
        out = preprocessor((x_in, y_in, sw_in))
        self.assertIsInstance(out, tuple)
        x, y, sw = out
        self.assertIn("vision_mask", x)
        self.assertAllEqual(y, y_in)
        self.assertAllEqual(sw, sw_in)

    def test_generate_preprocess_positional_sequence_length(self):
        # Same signature as `CausalLMPreprocessor.generate_preprocess`.
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            **self.init_text_kwargs
        )
        positional = preprocessor.generate_preprocess("the quick brown fox", 12)
        keyword = preprocessor.generate_preprocess(
            "the quick brown fox", sequence_length=12
        )
        self.assertEqual(positional["token_ids"].shape[-1], 12)
        self.assertAllEqual(positional["token_ids"], keyword["token_ids"])

    def test_flat_ragged_list_validation(self):
        # F6 Regression: flat ragged list with wrong prompt counts
        # should raise ValueError
        img = np.zeros([16, 16, 3])
        x = {
            "prompts": ["<|image|> a <|image|>", "b"],
            "images": [img, img],
        }
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Expected at least 2 images"):
            preprocessor(x)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello world", "this is testing"],
                "responses": [""],
            }
            self.text_preprocessor(input_data)
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello world", "this is testing"],
                "responses": ["hello", "", ""],
            }
            self.text_preprocessor(input_data)

    def test_text_input_dummy_pixel_values_shape(self):
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(**self.init_kwargs)
        model_patch_size = self.image_converter.model_patch_size
        expected_patch_dim = 3 * model_patch_size**2

        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        output = preprocessor(input_data)
        pixel_values = output[0]["pixel_values"]
        pixel_position_ids = output[0]["pixel_position_ids"]

        self.assertEqual(pixel_values.shape[1], 0)
        self.assertEqual(pixel_values.shape[3], expected_patch_dim)
        self.assertEqual(pixel_position_ids.shape[1], 0)
        self.assertEqual(pixel_position_ids.shape[3], 2)

    def test_video_preprocessor_basics(self):
        video_converter = Gemma4UnifiedVideoConverter(
            patch_size=4,
            num_frames=2,
            max_soft_tokens=2,
        )
        init_video_kwargs = {
            "tokenizer": self.tokenizer,
            "video_converter": video_converter,
            "sequence_length": 30,
            "num_frames_per_video": 2,
            "num_vision_tokens_per_frame": 2,
        }

        input_data = {
            "prompts": ["the quick brown fox <|video|>"],
            "responses": ["round"],
            "videos": np.ones((1, 2, 4, 4, 3), dtype="float32"),
        }

        preprocessor = Gemma4UnifiedCausalLMPreprocessor(**init_video_kwargs)
        output = preprocessor(input_data)

        self.assertIn("pixel_values", output[0])
        self.assertIn("pixel_position_ids", output[0])
        self.assertIn("vision_mask", output[0])

        import keras

        vision_mask = output[0]["vision_mask"]
        self.assertEqual(int(keras.ops.sum(vision_mask)), 2)

    def test_video_metadata_timestamps(self):
        """video_metadata attribute produces correct per-frame timestamps."""
        video_converter = Gemma4UnifiedVideoConverter(
            patch_size=4,
            num_frames=2,
            max_soft_tokens=2,
        )
        preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=100,
            num_frames_per_video=2,
            num_vision_tokens_per_frame=1,
            video_fps=1.0,
        )

        prompts = tf.constant(["<|video|>"])

        # === Default (no metadata): sequential [0, 1] at fps=1.0 ===
        expanded_default = preprocessor._expand_video_prompt(prompts, None)
        expanded_default_str = expanded_default.numpy()[0].decode("utf-8")
        self.assertIn("00:00", expanded_default_str)
        self.assertIn("00:01", expanded_default_str)

        # === With video_metadata: frames_indices=[0, 60], fps=1.0 ===
        preprocessor.video_metadata = [{"frames_indices": [0, 60], "fps": 1.0}]
        expanded_meta = preprocessor._expand_video_prompt(prompts, None)
        expanded_meta_str = expanded_meta.numpy()[0].decode("utf-8")
        self.assertIn("00:00", expanded_meta_str)
        self.assertIn("01:00", expanded_meta_str)
        self.assertNotIn("00:01", expanded_meta_str)

        # === Fallback restores default after clearing video_metadata ===
        preprocessor.video_metadata = None
        expanded_cleared = preprocessor._expand_video_prompt(prompts, None)
        self.assertEqual(
            expanded_cleared.numpy()[0], expanded_default.numpy()[0]
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        text_input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        vision_text_input_data = {
            "prompts": ["the quick brown fox <|image>"],
            "responses": ["round"],
            "pixel_values": np.ones((1, 2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((1, 2, 1, 2), dtype="int32"),
        }

        for preset in Gemma4UnifiedCausalLMPreprocessor.presets:
            if "_text" in preset:
                input_data = text_input_data
            else:
                input_data = vision_text_input_data
            self.run_preset_test(
                cls=Gemma4UnifiedCausalLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
