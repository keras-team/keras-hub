import keras
import pytest

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterPreprocessor,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class OpenAIPrivacyFilterPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ t", "Ġt h", "Ġth e"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["Ġquick", "Ġbrown", "Ġfox", "Ġjumps", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = OpenAIPrivacyFilterTokenizer(
            vocabulary=self.vocab, merges=self.merges
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
            cls=OpenAIPrivacyFilterPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_output_structure(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        output = preprocessor(["the quick brown fox"])
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)

    def test_sequence_length_override(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        output = preprocessor(["the quick brown fox"], sequence_length=16)
        x, _, _ = keras.utils.unpack_x_y_sample_weight(output)
        from keras import ops

        self.assertEqual(ops.shape(x["token_ids"])[-1], 16)

    def test_get_config(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        config = preprocessor.get_config()
        self.assertEqual(config["sequence_length"], 8)
        self.assertEqual(config["truncate"], "round_robin")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OpenAIPrivacyFilterPreprocessor.presets:
            self.run_preset_test(
                cls=OpenAIPrivacyFilterPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
