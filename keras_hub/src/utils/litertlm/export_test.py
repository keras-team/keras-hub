import importlib.util
import os
import types
import unittest
import unittest.mock

import keras
import numpy as np
import torch

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm import export
from keras_hub.src.utils.litertlm.adapter import _cpu_default_device_scope
from keras_hub.src.utils.litertlm.model_specs import GREEDY_SAMPLER_CONFIG
from keras_hub.src.utils.litertlm.model_specs import SamplerConfig

_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "LiteRT-LM export requires the PyTorch backend.",
)
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
class TestLiteRTLmExport(TestCase):
    def setUp(self):
        super().setUp()
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")
        self.tokenizer = GemmaTokenizer(proto=proto)
        self.backbone = GemmaBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            head_dim=8,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        self.preprocessor = GemmaCausalLMPreprocessor(
            tokenizer=self.tokenizer, sequence_length=8
        )
        self.model = GemmaCausalLM(
            backbone=self.backbone, preprocessor=self.preprocessor
        )
        self._set_random_weights(self.model)

    def _set_random_weights(self, model, seed=42):
        rng = np.random.default_rng(seed)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

    def _assert_kv_cache_close(
        self, keras_cache, tflite_out, atol, rtol, num_layers
    ):
        """Per-layer K/V parity between a Keras cache and TFLite outputs.

        ``num_layers`` is supplied by the caller rather than read off
        ``keras_cache`` so that a cache built with the wrong layer axis cannot
        turn the comparison loop into a no-op.
        """
        self.assertEqual(keras_cache.shape[1], num_layers)
        for i in range(num_layers):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_out[f"kv_cache_k_{i}"],
                atol=atol,
                rtol=rtol,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_out[f"kv_cache_v_{i}"],
                atol=atol,
                rtol=rtol,
            )

    def _call_with_cache_no_grad(
        self, model, tokens, cache, start_index, **kwargs
    ):
        """Eager `call_with_cache` under `no_grad`; numpy in, numpy out."""
        with torch.no_grad():
            logits, _, cache = model.call_with_cache(
                torch.from_numpy(tokens),
                torch.from_numpy(cache),
                start_index,
                **kwargs,
            )
        return logits.detach().cpu().numpy(), cache.detach().cpu().numpy()

    def test_export_tiny_gemma(self):
