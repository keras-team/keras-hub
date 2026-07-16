import numpy as np

from keras_hub.src.models.block_diffusion_lm_preprocessor import (
    BlockDiffusionLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class TestBlockDiffusionLMPreprocessor(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()
        self.sequence_length = 8
        self.canvas_length = 4
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": self.sequence_length,
            "canvas_length": self.canvas_length,
        }
        self.preprocessor = BlockDiffusionLMPreprocessor(**self.init_kwargs)

    def test_preset_accessors(self):
        subclass_presets = set(
            Gemma4BlockDiffusionLMPreprocessor.presets.keys()
        )
        all_presets = set(BlockDiffusionLMPreprocessor.presets.keys())
        self.assertTrue(subclass_presets.issubset(all_presets))

    def test_call_returns_correct_structure(self):
        x, y, sw = self.preprocessor(
            ["the quick brown fox", "the quick brown fox"]
        )
        seq_len = self.sequence_length
        self.assertEqual(x["token_ids"].shape, (2, seq_len))
        self.assertEqual(x["padding_mask"].shape, (2, seq_len))
        self.assertEqual(y.shape, (2, seq_len))
        self.assertEqual(sw.shape, (2, seq_len))

    def test_call_token_values(self):
        x, y, sw = self.preprocessor(["the quick brown fox"])
        # Vocab: 1=<bos>, 9=the, 14=quick, 10=brown, 12=fox, 2=<eos>, 0=<pad>
        expected_token_ids = [1, 9, 14, 10, 12, 2, 0, 0]
        expected_padding_mask = [1, 1, 1, 1, 1, 1, 0, 0]
        self.assertAllEqual(x["token_ids"][0], expected_token_ids)
        self.assertAllEqual(x["padding_mask"][0], expected_padding_mask)

    def test_call_labels_are_shifted(self):
        x, y, sw = self.preprocessor(["the quick brown fox"])
        # y is token_ids[1:], so first label is "the" (9)
        self.assertAllEqual(y[0, 0], 9)

    def test_call_no_start_end_token(self):
        preprocessor = BlockDiffusionLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=self.sequence_length,
            canvas_length=self.canvas_length,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(["the quick brown fox"] * 2)
        # Without BOS the first token is "the" (9).
        self.assertAllEqual(x["token_ids"][0, 0], 9)
        # Without EOS the sequence ends directly with pad (0).
        expected_token_ids = [9, 14, 10, 12, 0, 0, 0, 0]
        self.assertAllEqual(x["token_ids"][0], expected_token_ids)

    def test_generate_preprocess_appends_canvas(self):
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        expected_length = self.sequence_length + self.canvas_length
        self.assertAllEqual(output["token_ids"].shape[-1], expected_length)
        self.assertAllEqual(output["padding_mask"].shape[-1], expected_length)

    def test_generate_preprocess_prompt_token_values(self):
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        token_ids = np.array(output["token_ids"])
        # BOS + tokens, padded to sequence_length, then canvas zeros.
        # Prompt slice: [1, 9, 14, 10, 12, 0, 0, 0]
        expected_prompt = [1, 9, 14, 10, 12, 0, 0, 0]
        self.assertAllEqual(
            token_ids[..., : self.sequence_length], expected_prompt
        )

    def test_generate_preprocess_canvas_all_zeros(self):
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        token_ids = np.array(output["token_ids"])
        canvas_slice = token_ids[..., self.sequence_length :]
        pad_id = self.tokenizer.pad_token_id
        self.assertAllEqual(canvas_slice, np.full_like(canvas_slice, pad_id))

    def test_generate_preprocess_canvas_mask_all_false(self):
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        padding_mask = np.array(output["padding_mask"])
        canvas_mask = padding_mask[..., self.sequence_length :]
        self.assertAllEqual(canvas_mask, np.zeros_like(canvas_mask))

    def test_generate_postprocess_single(self):
        # 9=the, 14=quick, 10=brown, 12=fox, 0=<pad> (stripped)
        canvas = np.array([9, 14, 10, 12, 0, 0, 0, 0], dtype="int32")
        result = self.preprocessor.generate_postprocess(canvas)
        self.assertAllEqual(result, "the quick brown fox")

    def test_generate_postprocess_batch(self):
        canvas = np.array(
            [[9, 14, 10, 12, 0, 0], [9, 14, 0, 0, 0, 0]],
            dtype="int32",
        )
        results = self.preprocessor.generate_postprocess(canvas)
        self.assertEqual(len(results), 2)
        self.assertAllEqual(results[0], "the quick brown fox")
        self.assertAllEqual(results[1], "the quick")

    def test_serialization(self):
        self.run_serialization_test(self.preprocessor)

    def test_preprocessing_layer(self):
        self.run_preprocessing_layer_test(
            cls=BlockDiffusionLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=["the quick brown fox", "the quick brown fox"],
        )

    def test_call_sample_weight_values(self):
        _, _, sw = self.preprocessor(["the quick brown fox"])
        # sw=1 at every real label position, 0 at pad positions.
        # Packed (seq+1=9): [1,9,14,10,12,2,0,0,0] -> padding_mask[1:]:
        # [1,1,1,1,1,0,0,0]
        self.assertAllEqual(sw[0], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess_batched(self):
        output = self.preprocessor.generate_preprocess(
            ["the quick brown fox", "the quick brown fox"]
        )
        expected_length = self.sequence_length + self.canvas_length
        self.assertEqual(output["token_ids"].shape[0], 2)
        self.assertEqual(output["token_ids"].shape[1], expected_length)

    def test_call_truncates_long_input(self):
        # Input that tokenizes to more tokens than sequence_length must be
        # truncated to exactly sequence_length in the output.
        long_input = "the quick brown fox the earth is round" * 4
        x, y, sw = self.preprocessor([long_input, long_input])
        self.assertEqual(x["token_ids"].shape, (2, self.sequence_length))
        self.assertEqual(y.shape, (2, self.sequence_length))
        self.assertEqual(sw.shape, (2, self.sequence_length))

    def test_config_contains_canvas_length(self):
        config = self.preprocessor.get_config()
        self.assertIn("canvas_length", config)
        self.assertEqual(config["canvas_length"], self.canvas_length)

    def test_config_contains_sequence_length(self):
        config = self.preprocessor.get_config()
        self.assertIn("sequence_length", config)
        self.assertEqual(config["sequence_length"], self.sequence_length)

    def test_sequence_length_setter_updates_packer(self):
        self.preprocessor.build(None)  # ensure packer is initialized
        self.preprocessor.sequence_length = 16
        self.assertEqual(self.preprocessor.sequence_length, 16)
        self.assertEqual(self.preprocessor.packer.sequence_length, 16)
