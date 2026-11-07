from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.models.hrm_text.hrm_text_tokenizer_test import (
    make_tokenizer_assets,
)
from keras_hub.src.tests.test_case import TestCase


class HrmTextCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        vocabulary, merges = make_tokenizer_assets()
        self.tokenizer = HrmTextTokenizer(vocabulary=vocabulary, merges=merges)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 7,
        }

    def test_causal_preprocessor(self):
        self.run_preprocessor_test(
            cls=HrmTextCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=[" airplane at airport"],
            expected_output=(
                {
                    "token_ids": [[3, 27, 18, 28, 27, 20, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1]],
                    "token_type_ids": [[0, 0, 0, 0, 0, 0, 0]],
                },
                [[27, 18, 28, 27, 20, 1, 2]],
                [[1, 1, 1, 1, 1, 1, 0]],
            ),
        )

    def test_prefix_lm_preprocessor(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        outputs = preprocessor(
            {"prefix": [" airplane"], "response": [" at airport"]}
        )
        inputs, labels, weights = outputs
        self.assertAllEqual(inputs["token_ids"], [[3, 27, 18, 1, 28, 27, 1]])
        self.assertAllEqual(inputs["padding_mask"], [[1, 1, 1, 1, 1, 1, 1]])
        self.assertAllEqual(inputs["token_type_ids"], [[1, 1, 1, 1, 0, 0, 0]])
        self.assertAllEqual(labels, [[27, 18, 1, 28, 27, 1, 2]])
        self.assertAllEqual(weights, [[0, 0, 0, 1, 1, 1, 0]])

    def test_empty_prefix(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {"prefix": [""], "response": [" airplane"]}
        )
        self.assertAllEqual(inputs["token_type_ids"], [[1, 1, 0, 0, 0, 0, 0]])
        self.assertAllEqual(weights, [[0, 1, 1, 1, 0, 0, 0]])

    def test_empty_response(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {"prefix": [" airplane"], "response": [""]}
        )
        self.assertAllEqual(inputs["token_type_ids"], [[1, 1, 1, 1, 0, 0, 0]])
        self.assertAllEqual(weights, [[0, 0, 0, 1, 0, 0, 0]])

    def test_mixed_length_batch(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {
                "prefix": [" airplane", ""],
                "response": [" at airport", " airplane"],
            }
        )
        self.assertAllEqual(inputs["token_ids"].shape, (2, 7))
        self.assertAllEqual(weights.shape, (2, 7))

    def test_generate_round_trip(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        formatted = preprocessor.format_instruction(" airplane")
        self.assertEqual(
            formatted, "<|object_ref_start|> airplane<|im_end|>"
        )
        inputs = preprocessor.generate_preprocess(formatted)
        self.assertAllEqual(inputs["token_type_ids"], inputs["padding_mask"])
        self.assertAllEqual(
            inputs["token_ids"],
            [
                3,
                self.tokenizer.direct_condition_token_id,
                27,
                18,
                self.tokenizer.prefix_end_token_id,
                2,
                2,
            ],
        )
        self.assertEqual(
            preprocessor.generate_postprocess(inputs), " airplane"
        )

    def test_format_instruction_rejects_unknown_condition(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Unknown HRM-Text condition"):
            preprocessor.format_instruction(" airplane", "unknown")
