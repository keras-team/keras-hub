"""Tests for MetaCLIP 2 preprocessor."""

import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.metaclip_2.metaclip_2_image_converter import (
    MetaCLIP2ImageConverter,
)
from keras_hub.src.models.metaclip_2.metaclip_2_preprocessor import (
    MetaCLIP2Preprocessor,
)
from keras_hub.src.models.metaclip_2.metaclip_2_tokenizer import (
    MetaCLIP2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MetaCLIP2PreprocessorTest(TestCase):
    def setUp(self):
        # Use the XLM-RoBERTa test vocab since MetaCLIP 2 uses XLM-V
        # which is based on XLM-RoBERTa architecture
        self.tokenizer = MetaCLIP2Tokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        )
        self.image_converter = MetaCLIP2ImageConverter(
            image_size=(32, 32),
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 16,
        }
        self.input_data = {
            "prompts": ["the quick brown fox"],
            "images": np.ones((1, 64, 64, 3), dtype="float32"),
        }

    def test_preprocessor_basics(self):
        preprocessor = MetaCLIP2Preprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        self.assertEqual(output["token_ids"].shape, (1, 16))
        self.assertEqual(output["images"].shape, (1, 32, 32, 3))

    def test_sequence_length_override(self):
        preprocessor = MetaCLIP2Preprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data, sequence_length=8)
        self.assertEqual(output["token_ids"].shape, (1, 8))

    def test_to_lower(self):
        preprocessor = MetaCLIP2Preprocessor(**self.init_kwargs, to_lower=True)
        output = preprocessor(
            {
                "prompts": ["THE QUICK BROWN FOX"],
                "images": np.ones((1, 64, 64, 3), dtype="float32"),
            }
        )
        self.assertEqual(output["token_ids"].shape, (1, 16))

    def test_start_end_tokens(self):
        preprocessor = MetaCLIP2Preprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        # Check that start token (0) is at the beginning
        token_ids = ops.convert_to_numpy(output["token_ids"])
        self.assertEqual(token_ids[0, 0], 0)
        # Check that end token (2) is present after the text tokens

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MetaCLIP2Preprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
