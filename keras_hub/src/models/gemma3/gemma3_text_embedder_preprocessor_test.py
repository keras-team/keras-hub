import os

from keras_hub.src.models.gemma3.gemma3_text_embedder_preprocessor import (
    Gemma3TextEmbedderPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3TextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Gemma3Tokenizer(
            # Generated using create_gemma3_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            ),
            has_vision_tokens=False,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["the quick brown fox"]

    def test_output_keys(self):
        preprocessor = Gemma3TextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertNotIn("segment_ids", output)

    def test_padding_mask_matches_non_pad_tokens(self):
        preprocessor = Gemma3TextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        pad_id = self.tokenizer.pad_token_id
        expected_mask = output["token_ids"] != pad_id
        self.assertAllEqual(output["padding_mask"], expected_mask)

    def test_sequence_length(self):
        preprocessor = Gemma3TextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        self.assertEqual(output["token_ids"].shape[-1], 8)
        self.assertEqual(output["padding_mask"].shape[-1], 8)

    def test_bos_and_eos_present(self):
        preprocessor = Gemma3TextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        token_ids = output["token_ids"]
        # First token should be BOS.
        self.assertEqual(int(token_ids[0, 0]), self.tokenizer.start_token_id)
        # Find last real (non-pad) position; that token should be EOS.
        padding_mask = output["padding_mask"][0]
        real_len = int(sum(int(x) for x in padding_mask))
        self.assertEqual(
            int(token_ids[0, real_len - 1]), self.tokenizer.end_token_id
        )

    def test_sequence_length_truncation(self):
        preprocessor = Gemma3TextEmbedderPreprocessor(
            tokenizer=self.tokenizer, sequence_length=4
        )
        output = preprocessor(self.input_data)
        self.assertEqual(output["token_ids"].shape[-1], 4)
