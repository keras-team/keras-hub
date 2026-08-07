import numpy as np
import pytest

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm_preprocessor import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLMPreprocessor,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaBlockDiffusionLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # Text-only preprocessor (no media converters).
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
            "canvas_length": 4,
        }
        self.preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            **self.init_kwargs
        )

    def test_preprocessor_call_shape(self):
        output = self.run_preprocessing_layer_test(
            cls=DiffusionGemmaBlockDiffusionLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=["the quick brown fox", "the quick brown fox"],
            return_output=True,
        )
        x, y, sw = output
        # Verify shapes.
        self.assertEqual(x["token_ids"].shape[-1], 8)
        self.assertEqual(x["padding_mask"].shape[-1], 8)
        self.assertEqual(y.shape[-1], 8)
        self.assertEqual(sw.shape[-1], 8)
        # Verify token values for the first sample.
        # Vocab: 1=<bos>, 9=the, 14=quick, 10=brown, 12=fox, 2=<eos>, 0=<pad>
        self.assertAllEqual(x["token_ids"][0], [0, 0, 0, 1, 9, 14, 10, 12])
        self.assertAllEqual(x["padding_mask"][0], [0, 0, 0, 1, 1, 1, 1, 1])
        # y is token_ids shifted left by one.
        self.assertAllEqual(y[0], [0, 0, 1, 9, 14, 10, 12, 2])
        # sw is 1 for non-pad label positions.
        self.assertAllEqual(sw[0], [0, 0, 1, 1, 1, 1, 1, 1])

    def test_no_start_end_token(self):
        preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        input_data = ["the quick brown fox"] * 2
        x, y, sw = preprocessor(input_data)
        # Without BOS the first non-padding token should be "the" (id=9).
        self.assertAllEqual(x["token_ids"][0, -3], 9)

    def test_generate_preprocess_prompt_only(self):
        # token_ids should contain only the packed prompt, not the canvas.
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        self.assertAllEqual(output["token_ids"], [0, 0, 0, 1, 9, 14, 10, 12])
        self.assertAllEqual(output["padding_mask"], [0, 0, 0, 1, 1, 1, 1, 1])

    def test_generate_preprocess_batched(self):
        output = self.preprocessor.generate_preprocess(
            {"prompts": ["the", "the quick brown fox"]}
        )
        self.assertAllEqual(
            output["token_ids"],
            [[0, 0, 0, 0, 0, 0, 1, 9], [0, 0, 0, 1, 9, 14, 10, 12]],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]],
        )

    def test_generate_postprocess(self):
        # canvas_length=4; vocab: 9=the, 14=quick, 10=brown, 12=fox
        canvas = np.array([9, 14, 10, 12], dtype="int32")
        result = self.preprocessor.generate_postprocess(canvas)
        self.assertAllEqual(result, "the quick brown fox")

    def test_generate_postprocess_batched(self):
        # canvas_length=4; each row is one generated canvas.
        canvas = np.array(
            [[9, 14, 10, 12], [9, 14, 10, 12]],
            dtype="int32",
        )
        results = self.preprocessor.generate_postprocess(canvas)
        self.assertEqual(len(results), 2)

    def test_serialization(self):
        self.run_serialization_test(self.preprocessor)

    def test_sequence_length_setter(self):
        self.preprocessor.sequence_length = 16
        self.assertEqual(self.preprocessor.sequence_length, 16)
        # A subsequent call must produce outputs matching the new length.
        x, y, sw = self.preprocessor(["the quick brown fox"])
        self.assertEqual(x["token_ids"].shape[-1], 16)
        self.assertEqual(y.shape[-1], 16)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        for preset in DiffusionGemmaBlockDiffusionLMPreprocessor.presets:
            self.run_preset_test(
                cls=DiffusionGemmaBlockDiffusionLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
