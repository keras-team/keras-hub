import pytest

from keras_hub.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.bart.bart_tokenizer import BartTokenizer
from keras_hub.src.tests.test_case import TestCase


class BartSeq2SeqLMPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<s>", "<pad>", "</s>", "<mask>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = BartTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "encoder_sequence_length": 5,
            "decoder_sequence_length": 8,
        }
        self.input_data = (
            {
                "encoder_text": [" airplane at airport"],
                "decoder_text": [" airplane airport"],
            },
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=BartSeq2SeqLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "encoder_token_ids": [[3, 27, 18, 28, 0]],
                    "encoder_padding_mask": [[1, 1, 1, 1, 1]],
                    "decoder_token_ids": [[0, 3, 27, 18, 27, 20, 0, 2]],
                    "decoder_padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [[3, 27, 18, 27, 20, 0, 2, 2]],
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
            ),
            token_id_key="decoder_token_ids",
        )

    def test_generate_preprocess(self):
        preprocessor = BartSeq2SeqLMPreprocessor(**self.init_kwargs)
        input_data = {
            "encoder_text": [" airplane at airport"],
            "decoder_text": [" airplane airport"],
        }
        output = preprocessor.generate_preprocess(input_data)
        self.assertAllClose(
            output,
            {
                "encoder_token_ids": [[3, 27, 18, 28, 0]],
                "encoder_padding_mask": [[1, 1, 1, 1, 1]],
                "decoder_token_ids": [[0, 3, 27, 18, 27, 20, 2, 2]],
                "decoder_padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
            },
        )

    def test_generate_postprocess(self):
        preprocessor = BartSeq2SeqLMPreprocessor(**self.init_kwargs)
        input_data = {
            "decoder_token_ids": [3, 27, 18, 28, 0],
            "decoder_padding_mask": [1, 1, 1, 1, 1],
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(output, " airplane at")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BartSeq2SeqLMPreprocessor.presets:
            self.run_preset_test(
                cls=BartSeq2SeqLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
