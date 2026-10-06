import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifier,
)
from keras_hub.src.models.token_classifier import TokenClassifier
from keras_hub.src.tests.test_case import TestCase

TINY_PRESET = "hf://yujiepan/openai-privacy-filter-tiny-random"


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = OpenAIPrivacyFilterTokenClassifier.from_preset(TINY_PRESET)
        logits = model.predict(["My name is John Smith."])
        # One logit per BIOES label in the checkpoint's `id2label`.
        self.assertEqual(logits.shape[-1], 33)

    @pytest.mark.large
    def test_class_detection(self):
        model = TokenClassifier.from_preset(TINY_PRESET, load_weights=False)
        self.assertIsInstance(model, OpenAIPrivacyFilterTokenClassifier)
        model = Backbone.from_preset(TINY_PRESET, load_weights=False)
        self.assertIsInstance(model, OpenAIPrivacyFilterBackbone)

    @pytest.mark.large
    def test_logits_match_hf(self):
        # Build a small random HF model, convert it, and compare the logits.
        # The sliding window is shorter than the input and the second row is
        # padded, so both masks are exercised.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "OpenAIPrivacyFilterConfig"):
            pytest.skip("Needs a transformers release with this model.")
        config = transformers.OpenAIPrivacyFilterConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=8,
            num_experts_per_tok=2,
            sliding_window=3,
            initializer_range=0.2,
            num_labels=33,
            pad_token_id=0,
            eos_token_id=0,
        )
        torch.manual_seed(0)
        hf_model = transformers.OpenAIPrivacyFilterForTokenClassification(
            config
        ).eval()
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            model = OpenAIPrivacyFilterTokenClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        token_ids = np.array(
            [list(range(1, 13)), list(range(20, 28)) + [0] * 4], dtype="int32"
        )
        padding_mask = (np.arange(12) < np.array([[12], [8]])).astype("int32")
        logits = ops.convert_to_numpy(
            model({"token_ids": token_ids, "padding_mask": padding_mask})
        )
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=torch.tensor(token_ids),
                attention_mask=torch.tensor(padding_mask),
            ).logits.numpy()
        real = padding_mask.astype(bool)
        self.assertAllClose(logits[real], hf_logits[real], atol=1e-4)
