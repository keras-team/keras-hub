import pytest

from keras_hub.src.models.modern_bert.modern_bert_text_embedder_preprocessor import (  # noqa: E501
    ModernBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.modern_bert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        # Byte-level BPE vocabulary, as in the text classifier tests.
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]
        vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab))
        # `[CLS]` = 29, `[SEP]` = 30, `[PAD]` = 31.
        vocab += ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
        self.vocab = {token: i for i, token in enumerate(vocab)}

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 10,
        }
        self.input_data = (
            [" airplane at airport"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=ModernBertTextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    # [CLS] Ġair plane Ġat Ġair port [SEP] [PAD] [PAD] [PAD]
                    "token_ids": [[29, 23, 14, 24, 23, 16, 30, 31, 31, 31]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_sentence_pair(self):
        """Pairs are packed as `[CLS] A [SEP] B [SEP]`, as in Hugging Face."""
        preprocessor = ModernBertTextEmbedderPreprocessor(**self.init_kwargs)
        # Pass `y` explicitly so the tuple is read as two segments, rather
        # than being unpacked as `(x, y)`.
        x = preprocessor(([" airplane at"], [" airport"]), y=None)
        self.assertAllEqual(
            x["token_ids"], [[29, 23, 14, 24, 30, 23, 16, 30, 31, 31]]
        )
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

    def test_truncation_keeps_boundary_tokens(self):
        preprocessor = ModernBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=4,
        )
        x = preprocessor([" airplane at airport"])
        # [CLS] Ġair plane [SEP]
        self.assertAllEqual(x["token_ids"], [[29, 23, 14, 30]])
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1]])

    def test_no_segment_ids_in_output(self):
        preprocessor = ModernBertTextEmbedderPreprocessor(**self.init_kwargs)
        x, _, _ = preprocessor(*self.input_data)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertNotIn("segment_ids", x)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertTextEmbedderPreprocessor.presets:
            self.run_preset_test(
                cls=ModernBertTextEmbedderPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
