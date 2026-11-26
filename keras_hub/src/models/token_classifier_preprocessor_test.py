import re

import pytest
from keras import ops

from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifierPreprocessor,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.token_classifier_preprocessor import _SHORT_Y_MESSAGE
from keras_hub.src.models.token_classifier_preprocessor import (
    TokenClassifierPreprocessor,
)
from keras_hub.src.tests.mocks.mock_bpe_vocabulary import get_bpe_fixture
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import BUILTIN_PRESETS_FOR_BACKBONE

SHORT_Y = re.escape(_SHORT_Y_MESSAGE)


class TestTokenClassifierPreprocessor(TestCase):
    def setUp(self):
        vocab, merges = get_bpe_fixture()
        self.init_kwargs = {
            "tokenizer": OpenAIPrivacyFilterTokenizer(vocab, merges),
            "sequence_length": 8,
        }

    def test_preprocessor_basics(self):
        input_data = (
            ["airplane at airport"],
            [[1, 2, 0, 0, 0]],
            [[1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        self.run_preprocessor_test(
            cls=TokenClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
        )
        preprocessor = TokenClassifierPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, SHORT_Y):
            preprocessor(["airplane at airport"], [[1]])
        with self.assertRaisesRegex(ValueError, "Multi-segment"):
            preprocessor(x=(["airplane at airport"], ["airport"]))

    def test_sequence_length_setter(self):
        preprocessor = TokenClassifierPreprocessor(**self.init_kwargs)
        preprocessor(["airplane at airport"], [[1, 2, 0, 0, 0]])
        preprocessor.sequence_length = 12
        self.assertEqual(preprocessor.sequence_length, 12)
        self.assertEqual(preprocessor.packer.sequence_length, 12)
        self.assertEqual(preprocessor.y_packer.sequence_length, 12)
        x, y, sw = preprocessor(["airplane at airport"], [[1, 2, 0, 0, 0]])
        self.assertEqual(ops.shape(x["token_ids"]), (1, 12))
        self.assertEqual(ops.shape(y), (1, 12))
        self.assertEqual(ops.shape(sw), (1, 12))

    def test_preset_accessors(self):
        had_backbone = (
            OpenAIPrivacyFilterBackbone in BUILTIN_PRESETS_FOR_BACKBONE
        )
        BUILTIN_PRESETS_FOR_BACKBONE.setdefault(
            OpenAIPrivacyFilterBackbone, {}
        )["mock_privacy_filter"] = {}
        try:
            sub_presets = set(
                OpenAIPrivacyFilterTokenClassifierPreprocessor.presets.keys()
            )
            all_presets = set(TokenClassifierPreprocessor.presets.keys())
            gpt2_presets = set(GPT2Tokenizer.presets.keys())
            self.assertIn("mock_privacy_filter", sub_presets)
            self.assertIn("mock_privacy_filter", all_presets)
            self.assertTrue(sub_presets.issubset(all_presets))
            self.assertTrue(gpt2_presets.isdisjoint(all_presets))
        finally:
            BUILTIN_PRESETS_FOR_BACKBONE[OpenAIPrivacyFilterBackbone].pop(
                "mock_privacy_filter", None
            )
            if (
                not had_backbone
                and not BUILTIN_PRESETS_FOR_BACKBONE[
                    OpenAIPrivacyFilterBackbone
                ]
            ):
                BUILTIN_PRESETS_FOR_BACKBONE.pop(
                    OpenAIPrivacyFilterBackbone, None
                )

    @pytest.mark.large
    def test_from_preset(self):
        preprocessor = TokenClassifierPreprocessor.from_preset(
            "hf://yujiepan/openai-privacy-filter-tiny-random",
            sequence_length=16,
        )
        self.assertIsInstance(
            preprocessor, OpenAIPrivacyFilterTokenClassifierPreprocessor
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            OpenAIPrivacyFilterTokenClassifierPreprocessor.from_preset(
                "gpt2_base_en"
            )
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            OpenAIPrivacyFilterTokenClassifierPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
