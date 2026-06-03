import numpy as np
import pytest

from keras_hub.src.models.bge.bge_text_embedder_preprocessor import (
    BgeTextEmbedderPreprocessor,
)
from keras_hub.src.models.bge.bge_tokenizer import BgeTokenizer
from keras_hub.src.tests.test_case import TestCase


class BgeTextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.tokenizer = BgeTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["the quick brown fox."]

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=BgeTextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_segment_ids_present(self):
        """Embedder preprocessor must include segment_ids."""
        preprocessor = BgeTextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(self.input_data)
        self.assertIn("segment_ids", output)

    def test_segment_ids_all_zeros_single_sentence(self):
        """Single-sentence input should have all-zero segment_ids."""
        preprocessor = BgeTextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(["the quick brown fox."])
        seg_np = np.array(output["segment_ids"])
        mask_np = np.array(output["padding_mask"])
        self.assertTrue((seg_np[mask_np == 1] == 0).all())

    def test_cls_token_at_position_zero(self):
        """[CLS] token (id=2 in test vocab) must be at position 0."""
        preprocessor = BgeTextEmbedderPreprocessor(**self.init_kwargs)
        output = preprocessor(["the quick brown fox."])
        token_ids = np.array(output["token_ids"])
        cls_id = self.tokenizer.cls_token_id
        self.assertEqual(token_ids[0, 0], cls_id)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BgeTextEmbedderPreprocessor,
            preset="bge_small_en_v1.5",
            input_data=["I love machine learning and nlp"],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BgeTextEmbedderPreprocessor.presets:
            self.run_preset_test(
                cls=BgeTextEmbedderPreprocessor,
                preset=preset,
                input_data=["The quick brown fox."],
            )
