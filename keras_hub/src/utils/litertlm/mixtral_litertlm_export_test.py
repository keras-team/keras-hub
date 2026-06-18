import os

import numpy as np

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.models.mixtral.mixtral_causal_lm import MixtralCausalLM
from keras_hub.src.models.mixtral.mixtral_causal_lm_preprocessor import (
    MixtralCausalLMPreprocessor,
)
from keras_hub.src.models.mixtral.mixtral_tokenizer import MixtralTokenizer
from keras_hub.src.tests.test_case import TestCase


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
