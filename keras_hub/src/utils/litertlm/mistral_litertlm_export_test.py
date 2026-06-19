import os
import unittest

import numpy as np

try:
    import litert_torch
except ImportError:
    litert_torch = None

try:
    import litert_lm_builder
except ImportError:
    litert_lm_builder = None

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.models.mistral.mistral_causal_lm_preprocessor import (
    MistralCausalLMPreprocessor,
)
from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase


@unittest.skipIf(
    litert_torch is None,
    "LiteRT-LM export requires `litert-torch`. "
    "Install it with: pip install litert-torch",
)
@unittest.skipIf(
    litert_lm_builder is None,
    "LiteRT-LM export requires `litert-lm-builder`. "
    "Install it with: pip install litert-lm-builder",
)
class TestMistralLiteRTLmExport(TestCase):
    def test_mistral_litertlm_export_baked_in(self):
        proto = os.path.join(self.get_test_data_dir(), "mistral_test_vocab.spm")
        tokenizer = MistralTokenizer(proto=proto)

        backbone = MistralBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        preprocessor = MistralCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = MistralCausalLM(backbone=backbone, preprocessor=preprocessor)

        # Set random weights for determinism.
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

        input_data = np.array([[1, 2, 3, 4]], dtype=np.int32)
        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=4,
            verify_model_type="generic_model",
            verify_numerics=False,
            verify_generation=True,
            generation_max_tokens=4,
        )
