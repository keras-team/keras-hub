"""CPU unit tests for mapping KerasHub samplers to SamplingParams kwargs."""

import builtins
from unittest import mock

import keras
import numpy as np
import pytest

from keras_hub.src.samplers.beam_sampler import BeamSampler
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.samplers.random_sampler import RandomSampler
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import registry
from keras_hub.src.vllm.registry import _default_gpu_dtype
from keras_hub.src.vllm.registry import _default_sampling_kwargs
from keras_hub.src.vllm.registry import _normalize_dtype
from keras_hub.src.vllm.registry import sampler_to_sampling_kwargs


class DefaultGpuDtypeTest(TestCase):
    """The default must be one the GPU actually has."""

    def _with_capability(self, major, available=True):
        torch = pytest.importorskip("torch")
        self.addCleanup(
            setattr, torch.cuda, "is_available", torch.cuda.is_available
        )
        self.addCleanup(
            setattr,
            torch.cuda,
            "get_device_capability",
            torch.cuda.get_device_capability,
        )
        torch.cuda.is_available = lambda: available
        torch.cuda.get_device_capability = lambda *a, **kw: (major, 0)

    def test_pre_ampere_gets_float16(self):
        # A T4 is 7.5 and has no bfloat16 at all.
        self._with_capability(7)
        self.assertEqual(_default_gpu_dtype(), "float16")

    def test_ampere_and_later_get_bfloat16(self):
        self._with_capability(8)
        self.assertEqual(_default_gpu_dtype(), "bfloat16")

    def test_no_gpu_falls_back_to_bfloat16(self):
        self._with_capability(7, available=False)
        self.assertEqual(_default_gpu_dtype(), "bfloat16")

    def test_without_torch_falls_back_to_bfloat16(self):
        # Importing this module must not require torch.
        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", no_torch):
            self.assertEqual(registry._default_gpu_dtype(), "bfloat16")


class SamplerDefaultsTest(TestCase):
    def test_normalize_dtype_forms(self):
        self.assertEqual(_normalize_dtype(None), "bfloat16")
        self.assertEqual(_normalize_dtype("auto"), "bfloat16")
        self.assertEqual(_normalize_dtype("float32"), "float32")
        self.assertEqual(_normalize_dtype(np.float32), "float32")
        self.assertEqual(
            _normalize_dtype(keras.DTypePolicy("mixed_bfloat16")), "bfloat16"
        )

    def test_default_kwargs_read_without_model_build(self):
        # KerasHub's compile default is top_k(k=5, temperature=1.0); read
        # from the signature, never by building a model.
        self.assertEqual(
            _default_sampling_kwargs(), {"top_k": 5, "temperature": 1.0}
        )

    def test_greedy_maps_to_temperature_zero(self):
        self.assertEqual(
            sampler_to_sampling_kwargs(GreedySampler()), {"temperature": 0.0}
        )

    def test_top_k_maps_k_and_temperature(self):
        kwargs = sampler_to_sampling_kwargs(TopKSampler(k=7, temperature=0.7))
        self.assertEqual(kwargs, {"temperature": 0.7, "top_k": 7})

    def test_top_k_defaults(self):
        # CausalLM.compile default is sampler="top_k" (k=5, temperature=1.0).
        kwargs = sampler_to_sampling_kwargs(TopKSampler())
        self.assertEqual(kwargs, {"temperature": 1.0, "top_k": 5})

    def test_top_p_maps_p_and_optional_k(self):
        kwargs = sampler_to_sampling_kwargs(TopPSampler(p=0.9))
        self.assertEqual(kwargs, {"temperature": 1.0, "top_p": 0.9})
        kwargs = sampler_to_sampling_kwargs(TopPSampler(p=0.9, k=40))
        self.assertEqual(
            kwargs, {"temperature": 1.0, "top_p": 0.9, "top_k": 40}
        )

    def test_random_maps_to_temperature_only(self):
        kwargs = sampler_to_sampling_kwargs(RandomSampler(temperature=1.3))
        self.assertEqual(kwargs, {"temperature": 1.3})

    def test_seed_maps_when_set(self):
        kwargs = sampler_to_sampling_kwargs(TopKSampler(k=3, seed=7))
        self.assertEqual(kwargs["seed"], 7)
        # No seed set -> no seed key.
        self.assertNotIn("seed", sampler_to_sampling_kwargs(TopKSampler()))

    def test_unsupported_sampler_returns_none(self):
        self.assertIsNone(sampler_to_sampling_kwargs(BeamSampler()))
