import os

import pytest

from keras_hub.src.models.t5.t5_seq_2_seq_lm_preprocessor import (
    T5Seq2SeqLMPreprocessor,
)
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tests.test_case import TestCase


class T5Seq2SeqLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = T5Tokenizer(
            # Generated using create_t5_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
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
        self.run_preprocessing_layer_test(
            cls=T5Seq2SeqLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    # The encoder sequence has no start token, just `"</s>"`.
                    "encoder_token_ids": [[4, 9, 5, 7, 1, 0, 0, 0]],
                    "encoder_padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                    # The decoder sequence is shifted right by `"<pad>"`.
                    "decoder_token_ids": [[0, 4, 6, 8, 10, 1, 0, 0]],
                    "decoder_padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[4, 6, 8, 10, 1, 0, 0, 0]],  # Labels shifted left.
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Zero out unlabeled positions.
            ),
        )

    def test_generate_preprocess(self):
        input_data = {
            "encoder_text": "the quick brown fox",
            "decoder_text": "the earth",
        }
        preprocessor = T5Seq2SeqLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["encoder_token_ids"], [4, 9, 5, 7, 1, 0, 0, 0])
        self.assertAllEqual(x["encoder_padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])
        # No end token is added, as generation continues from the prompt.
        self.assertAllEqual(x["decoder_token_ids"], [0, 4, 6, 0, 0, 0, 0, 0])
        self.assertAllEqual(x["decoder_padding_mask"], [1, 1, 1, 0, 0, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "decoder_token_ids": [0, 4, 6, 8, 10, 1, 0, 0],
            "decoder_padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = T5Seq2SeqLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the earth is round")

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Seq2SeqLMPreprocessor.presets:
            self.run_preset_test(
                cls=T5Seq2SeqLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
