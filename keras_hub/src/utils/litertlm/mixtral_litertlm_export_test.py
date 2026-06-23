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

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.models.mixtral.mixtral_causal_lm import MixtralCausalLM
from keras_hub.src.models.mixtral.mixtral_causal_lm_preprocessor import (
    MixtralCausalLMPreprocessor,
)
from keras_hub.src.models.mixtral.mixtral_tokenizer import MixtralTokenizer
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
class TestMixtralLiteRTLmExport(TestCase):
    def test_mixtral_litertlm_export_baked_in(self):
        proto = os.path.join(self.get_test_data_dir(), "mixtral_test_vocab.spm")
        tokenizer = MixtralTokenizer(proto=proto)

        backbone = MixtralBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            num_experts=2,
            top_k=2,
        )
        preprocessor = MixtralCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = MixtralCausalLM(backbone=backbone, preprocessor=preprocessor)

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
