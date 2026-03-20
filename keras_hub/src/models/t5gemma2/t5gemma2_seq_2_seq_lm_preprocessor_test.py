import os

import numpy as np
import pytest

from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm_preprocessor import (
    T5Gemma2Seq2SeqLMPreprocessor,
)
from keras_hub.src.models.t5gemma2.t5gemma2_tokenizer import T5Gemma2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class T5Gemma2Seq2SeqLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = T5Gemma2Tokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "encoder_sequence_length": 8,
            "decoder_sequence_length": 8,
        }
        self.input_data = (
            {
                "encoder_text": ["the quick brown fox"],
                "decoder_text": ["the earth is round"],
            },
        )

    def test_preprocessor_basics(self):
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(**self.init_kwargs)
        output = preprocessor(*self.input_data)
        x, y, sample_weight = output

        # Verify output keys.
        self.assertIn("encoder_token_ids", x)
        self.assertIn("encoder_padding_mask", x)
        self.assertIn("decoder_token_ids", x)
        self.assertIn("decoder_padding_mask", x)

        # Verify shapes.
        self.assertEqual(x["encoder_token_ids"].shape[-1], 8)
        self.assertEqual(x["decoder_token_ids"].shape[-1], 8)

    def test_generate_preprocess(self):
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(**self.init_kwargs)
        input_data = {
            "encoder_text": ["the quick brown fox"],
            "decoder_text": ["the earth is round"],
        }
        output = preprocessor.generate_preprocess(input_data)
        self.assertIn("encoder_token_ids", output)
        self.assertIn("encoder_padding_mask", output)
        self.assertIn("decoder_token_ids", output)
        self.assertIn("decoder_padding_mask", output)

    def test_generate_postprocess(self):
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(**self.init_kwargs)
        input_data = {
            "decoder_token_ids": [2, 9, 14, 10, 1],
            "decoder_padding_mask": [1, 1, 1, 1, 1],
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertIsInstance(output, str)

    def test_add_vision_inputs_multimodal(self):
        """Multimodal preprocessor should add dummy vision inputs
        when text-only is used for inference."""
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(
            **self.init_kwargs,
            image_size=64,
            num_vision_tokens_per_image=16,
        )
        x = {
            "encoder_token_ids": np.ones((2, 8), dtype="int32"),
            "encoder_padding_mask": np.ones((2, 8), dtype="int32"),
            "decoder_token_ids": np.ones((2, 8), dtype="int32"),
            "decoder_padding_mask": np.ones((2, 8), dtype="int32"),
        }
        result = preprocessor._add_vision_inputs(x, batch_size=2)

        # Should add dummy images and vision indices.
        self.assertIn("images", result)
        self.assertIn("vision_indices", result)
        self.assertEqual(result["images"].shape, (2, 1, 64, 64, 3))
        self.assertEqual(result["vision_indices"].shape, (2, 16))
        # Dummy values should be all zeros.
        np.testing.assert_array_equal(
            result["images"], np.zeros_like(result["images"])
        )
        np.testing.assert_array_equal(
            result["vision_indices"],
            np.zeros_like(result["vision_indices"]),
        )

    def test_add_vision_inputs_skips_when_images_present(self):
        """Should not overwrite existing images."""
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(
            **self.init_kwargs,
            image_size=64,
            num_vision_tokens_per_image=16,
        )
        existing_images = np.ones((1, 1, 64, 64, 3), dtype="float32")
        x = {
            "encoder_token_ids": np.ones((1, 8), dtype="int32"),
            "images": existing_images,
        }
        result = preprocessor._add_vision_inputs(x, batch_size=1)
        np.testing.assert_array_equal(result["images"], existing_images)

    def test_serialization(self):
        preprocessor = T5Gemma2Seq2SeqLMPreprocessor(
            **self.init_kwargs,
            image_size=128,
            num_vision_tokens_per_image=32,
        )
        config = preprocessor.get_config()
        self.assertEqual(config["image_size"], 128)
        self.assertEqual(config["num_vision_tokens_per_image"], 32)
        self.assertEqual(config["add_start_token"], False)
        self.assertEqual(config["add_end_token"], True)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Gemma2Seq2SeqLMPreprocessor.presets:
            self.run_preset_test(
                cls=T5Gemma2Seq2SeqLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
