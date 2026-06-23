import importlib.util
import os
import unittest

import numpy as np

_LITERT_TORCH_AVAILABLE = (
    importlib.util.find_spec("litert_torch") is not None
)
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)

from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_hub.src.models.phi3.phi3_causal_lm import Phi3CausalLM
from keras_hub.src.models.phi3.phi3_causal_lm_preprocessor import (
    Phi3CausalLMPreprocessor,
)
from keras_hub.src.models.phi3.phi3_tokenizer import Phi3Tokenizer
from keras_hub.src.tests.test_case import TestCase


@unittest.skipIf(
    not _LITERT_TORCH_AVAILABLE,
    "LiteRT-LM export requires `litert-torch`. "
    "Install it with: pip install litert-torch",
)
@unittest.skipIf(
    not _LITERT_LM_BUILDER_AVAILABLE,
    "LiteRT-LM export requires `litert-lm-builder`. "
    "Install it with: pip install litert-lm-builder",
)
class TestPhi3LiteRTLmExport(TestCase):
    def _build_tiny_model(self):
        proto = os.path.join(self.get_test_data_dir(), "phi3_test_vocab.spm")
        tokenizer = Phi3Tokenizer(proto=proto)

        vocabulary_size = tokenizer.vocabulary_size()
        backbone = Phi3Backbone(
            vocabulary_size=vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            max_sequence_length=8,
        )
        preprocessor = Phi3CausalLMPreprocessor(
            tokenizer=tokenizer,
            sequence_length=8,
        )
        model = Phi3CausalLM(backbone=backbone, preprocessor=preprocessor)

        # Set random weights for determinism. Dummy weights mean output
        # values are meaningless, but the export and runtime smoke test
        # still exercise the full pipeline.
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

        return model

    def test_phi3_litertlm_export_baked_in(self):
        model = self._build_tiny_model()
        input_data = np.array([[1, 2, 3, 4]], dtype=np.int32)
        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=8,
            verify_model_type="generic_model",
            verify_numerics=False,
            verify_generation=True,
            generation_max_tokens=8,
        )
