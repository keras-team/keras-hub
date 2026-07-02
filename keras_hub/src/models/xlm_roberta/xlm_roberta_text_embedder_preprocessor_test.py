import os

import numpy as np
import pytest

from keras_hub.src.models.xlm_roberta.xlm_roberta_text_embedder_preprocessor import (  # noqa: E501
    XLMRobertaTextEmbedderPreprocessor,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class XLMRobertaTextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = XLMRobertaTokenizer(
            # Generated using create_xlm_roberta_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["the quick brown fox"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=XLMRobertaTextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[0, 6, 11, 7, 9, 2, 1, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_no_segment_ids_in_output(self):
        preprocessor = XLMRobertaTextEmbedderPreprocessor(**self.init_kwargs)
        output, _, _ = preprocessor(*self.input_data)
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertNotIn("segment_ids", output)

    def test_padding_mask_uses_pad_token_id_1(self):
        preprocessor = XLMRobertaTextEmbedderPreprocessor(**self.init_kwargs)
        output, _, _ = preprocessor(*self.input_data)
        token_ids = output["token_ids"]
        padding_mask = output["padding_mask"]
        # Where token_id is 1 (<pad>), mask should be 0.
        for t, m in zip(np.array(token_ids[0]), np.array(padding_mask[0])):
            if t == 1:
                self.assertEqual(m, 0)
            else:
                self.assertEqual(m, 1)

    def test_errors_for_2d_list_input(self):
        preprocessor = XLMRobertaTextEmbedderPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTextEmbedderPreprocessor.presets:
            self.run_preset_test(
                cls=XLMRobertaTextEmbedderPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
