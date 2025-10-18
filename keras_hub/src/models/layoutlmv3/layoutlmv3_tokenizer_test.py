import numpy as np
from keras import ops

from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import (
    LayoutLMv3Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class LayoutLMv3TokenizerTest(TestCase):
    def setUp(self):
        # Create a simple vocabulary for testing
        self.vocabulary = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "hello": 5,
            "world": 6,
            "how": 7,
            "are": 8,
            "you": 9,
            "good": 10,
            "morning": 11,
        }

        # Create simple merges for BPE
        self.merges = [
            "h e",
            "l l",
            "o w",
            "w o",
            "r l",
            "d </w>",
        ]

        self.tokenizer = LayoutLMv3Tokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
            sequence_length=16,
        )

    def test_tokenizer_basics(self):
        # Test basic properties
        self.assertEqual(self.tokenizer.cls_token, "[CLS]")
        self.assertEqual(self.tokenizer.sep_token, "[SEP]")
        self.assertEqual(self.tokenizer.pad_token, "[PAD]")
        self.assertEqual(self.tokenizer.mask_token, "[MASK]")
        self.assertEqual(self.tokenizer.unk_token, "[UNK]")

    def test_tokenizer_functionality(self):
        """Test tokenizer using the standardized test helper."""
        self.run_preprocessor_test(
            cls=LayoutLMv3Tokenizer,
            init_kwargs={
                "vocabulary": self.vocabulary,
                "merges": self.merges,
                "sequence_length": 16,
            },
            input_data="hello world",
        )

    def test_list_tokenization(self):
        # Test list of strings tokenization
        texts = ["hello world", "how are you"]
        output = self.tokenizer(texts)

        # Check shapes for batch processing
        self.assertEqual(output["token_ids"].shape, (2, 16))
        self.assertEqual(output["padding_mask"].shape, (2, 16))
        self.assertEqual(output["bbox"].shape, (2, 16, 4))

    def test_bbox_processing(self):
        # Test with bounding boxes provided
        texts = ["hello world"]
        bbox = [[[0, 0, 100, 50], [100, 0, 200, 50]]]

        output = self.tokenizer(texts, bbox=bbox)

        # Check that bbox was processed correctly
        self.assertEqual(output["bbox"].shape, (1, 16, 4))

        # Check that dummy bbox was added for special tokens
        bbox_values = ops.convert_to_numpy(output["bbox"][0])
        # First position should be dummy for [CLS]
        self.assertTrue(np.array_equal(bbox_values[0], [0, 0, 0, 0]))

    def test_bbox_expansion_for_subwords(self):
        # Test that bounding boxes are properly expanded for subword tokens
        texts = ["hello"]
        bbox = [[[0, 0, 100, 50]]]  # One bbox for one word

        output = self.tokenizer(texts, bbox=bbox)

        # The bbox should be expanded to cover all tokens including specials
        self.assertEqual(output["bbox"].shape, (1, 16, 4))

    def test_mismatched_bbox_count(self):
        # Test handling when bbox count doesn't match word count
        texts = ["hello world how"]  # 3 words
        bbox = [[[0, 0, 100, 50], [100, 0, 200, 50]]]  # 2 bboxes

        # Should handle gracefully by using dummy boxes
        output = self.tokenizer(texts, bbox=bbox)

        self.assertEqual(output["bbox"].shape, (1, 16, 4))

    def test_no_bbox_provided(self):
        # Test tokenization without bounding boxes
        texts = ["hello world"]
        output = self.tokenizer(texts)

        # Should create dummy bbox tensor
        self.assertEqual(output["bbox"].shape, (1, 16, 4))

        # All bbox values should be zeros (dummy)
        bbox_values = ops.convert_to_numpy(output["bbox"][0])
        for i in range(bbox_values.shape[0]):
            self.assertTrue(np.array_equal(bbox_values[i], [0, 0, 0, 0]))

    def test_get_config(self):
        config = self.tokenizer.get_config()

        # Check that all expected keys are in config
        expected_keys = [
            "vocabulary",
            "lowercase",
            "strip_accents",
            "split",
            "split_on_cjk",
            "suffix_indicator",
            "oov_token",
            "cls_token",
            "sep_token",
            "pad_token",
            "mask_token",
            "unk_token",
        ]

        for key in expected_keys:
            self.assertIn(key, config)

    def test_from_config(self):
        config = self.tokenizer.get_config()
        restored_tokenizer = LayoutLMv3Tokenizer.from_config(config)

        # Test that restored tokenizer works the same
        output1 = self.tokenizer("hello world")
        output2 = restored_tokenizer("hello world")

        self.assertAllClose(output1["token_ids"], output2["token_ids"])
        self.assertAllClose(output1["padding_mask"], output2["padding_mask"])

    def test_special_token_handling(self):
        # Test that special tokens are handled correctly
        texts = ["hello"]
        output = self.tokenizer(texts)

        token_ids = ops.convert_to_numpy(output["token_ids"][0])

        # Should start with [CLS] and end with [SEP]
        self.assertEqual(token_ids[0], self.vocabulary["[CLS]"])

        # Find the last non-padding token - should be [SEP]
        padding_mask = ops.convert_to_numpy(output["padding_mask"][0])
        last_token_idx = np.sum(padding_mask) - 1
        self.assertEqual(token_ids[last_token_idx], self.vocabulary["[SEP]"])

    def test_sequence_length_parameter(self):
        # Test with custom sequence length
        custom_tokenizer = LayoutLMv3Tokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
            sequence_length=8,
        )

        output = custom_tokenizer("hello world")

        # Check that output respects custom sequence length
        self.assertEqual(output["token_ids"].shape, (1, 8))
        self.assertEqual(output["padding_mask"].shape, (1, 8))
        self.assertEqual(output["bbox"].shape, (1, 8, 4))

    def test_padding_and_truncation(self):
        # Test with a very long input
        long_text = " ".join(["hello"] * 20)
        output = self.tokenizer(long_text)

        # Should be truncated to sequence_length
        self.assertEqual(output["token_ids"].shape, (1, 16))

        # Test with short input
        short_text = "hello"
        output = self.tokenizer(short_text)

        # Should be padded to sequence_length
        self.assertEqual(output["token_ids"].shape, (1, 16))

        # Check that padding tokens are used
        token_ids = ops.convert_to_numpy(output["token_ids"][0])
        padding_mask = ops.convert_to_numpy(output["padding_mask"][0])

        # Find first padding position
        padding_positions = np.where(padding_mask == 0)[0]
        if len(padding_positions) > 0:
            first_pad_pos = padding_positions[0]
            self.assertEqual(token_ids[first_pad_pos], self.vocabulary["[PAD]"])

    def test_batch_processing_consistency(self):
        # Test that batch processing gives same results as individual processing
        texts = ["hello world", "how are you"]

        # Process as batch
        batch_output = self.tokenizer(texts)

        # Process individually
        individual_outputs = []
        for text in texts:
            individual_outputs.append(self.tokenizer(text))

        # Compare results
        for i in range(len(texts)):
            batch_token_ids = ops.convert_to_numpy(batch_output["token_ids"])
            individual_token_ids = ops.convert_to_numpy(
                individual_outputs[i]["token_ids"]
            )
            self.assertAllClose(
                batch_token_ids[i : i + 1],
                individual_token_ids,
            )

            batch_padding_mask = ops.convert_to_numpy(
                batch_output["padding_mask"]
            )
            individual_padding_mask = ops.convert_to_numpy(
                individual_outputs[i]["padding_mask"]
            )
            self.assertAllClose(
                batch_padding_mask[i : i + 1],
                individual_padding_mask,
            )

    def test_empty_input(self):
        # Test handling of empty input
        output = self.tokenizer("")

        # Should still produce valid output with special tokens
        self.assertEqual(output["token_ids"].shape, (1, 16))
        self.assertEqual(output["padding_mask"].shape, (1, 16))
        self.assertEqual(output["bbox"].shape, (1, 16, 4))

        # Should contain [CLS] and [SEP] tokens
        token_ids = ops.convert_to_numpy(output["token_ids"][0])
        self.assertEqual(token_ids[0], self.vocabulary["[CLS]"])
        self.assertEqual(token_ids[1], self.vocabulary["[SEP]"])

    def test_oov_token_handling(self):
        # Test handling of out-of-vocabulary tokens
        output = self.tokenizer("unknown_token")

        # Should use [UNK] token for unknown words
        token_ids = ops.convert_to_numpy(output["token_ids"][0])

        # Check that [UNK] token appears (excluding [CLS] and [SEP])
        self.assertIn(self.vocabulary["[UNK]"], token_ids[1:-1])

    def test_case_sensitivity(self):
        # Test case handling based on lowercase parameter
        output1 = self.tokenizer("Hello")
        output2 = self.tokenizer("hello")

        # Should be the same if lowercase=True (default)
        self.assertAllClose(output1["token_ids"], output2["token_ids"])
