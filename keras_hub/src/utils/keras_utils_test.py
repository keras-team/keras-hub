import contextlib
import os
import sys
import types
from unittest import mock

import keras
import numpy as np
from absl.testing import parameterized
from keras.src.utils import module_utils

from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.image_to_image import ImageToImage
from keras_hub.src.models.inpaint import Inpaint
from keras_hub.src.models.text_to_image import TextToImage
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import (
    make_pluggable_backend_generate_function,
)


class CloneInitializerTest(TestCase):
    def test_config_equality(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer.get_config(), clone.get_config())

    def test_random_output(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertNotAllEqual(initializer(shape=(2, 2)), clone(shape=(2, 2)))

    def test_strings(self):
        initializer = "glorot_uniform"
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer, clone)


# Each task base class with the static arguments of its `generate_step`.
GENERATE_TASKS = (
    ("causal_lm", CausalLM, ("stop_token_ids",)),
    ("block_diffusion_lm", BlockDiffusionLM, ("max_length", "stop_token_ids")),
    ("text_to_image", TextToImage, ()),
    ("image_to_image", ImageToImage, ()),
    ("inpaint", Inpaint, ()),
)


def fake_generate_function(*args, **kwargs):
    """Stands in for the function a backend hook returns."""


def backend_package(name, with_keras_hub=True, **attributes):
    """Put a `keras_<name>` backend package into `sys.modules`."""
    package = types.ModuleType(f"keras_{name}")
    package.__path__ = []
    modules = {package.__name__: package}
    if with_keras_hub:
        src_module = types.ModuleType(f"keras_{name}.src")
        src_module.__path__ = []
        hub_module = types.ModuleType(f"keras_{name}.src.keras_hub")
        hub_module.__dict__.update(attributes)
        package.src = src_module
        src_module.keras_hub = hub_module
        modules[src_module.__name__] = src_module
        modules[hub_module.__name__] = hub_module
    else:
        package.__dict__.update(attributes)
    return mock.patch.dict(sys.modules, modules)


@contextlib.contextmanager
def backend_name(name, unofficial=True):
    """Make Keras report `name` as the active backend."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(keras.src.backend.config, "_BACKEND", name)
        )
        if unofficial:
            stack.enter_context(
                mock.patch.dict(
                    os.environ, {"KERAS_ALLOW_UNOFFICIAL_BACKEND": "1"}
                )
            )
        yield


class _BackendGenerateFunctionTestCase(TestCase):
    def setUp(self):
        super().setUp()
        built_in_backends = ("jax", "numpy", "tensorflow", "torch")
        if keras.config.backend() not in built_in_backends:
            self.skipTest("Simulates a pluggable backend on a built-in one.")
        self.hook = mock.Mock(return_value=fake_generate_function)


class PluggableBackendGenerateFunctionTest(_BackendGenerateFunctionTestCase):
    def setUp(self):
        super().setUp()
        if not hasattr(module_utils, "get_pluggable_backend_module"):
            self.skipTest(
                "`get_pluggable_backend_module` requires Keras >= 3.16."
            )

    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_backend_hook_is_used(self, cls, static_argnames):
        model = cls()
        with (
            backend_package("fakebackend", make_generate_function=self.hook),
            backend_name("fakebackend"),
        ):
            generate_function = model.make_generate_function()
            self.assertIs(generate_function, fake_generate_function)
            self.assertIs(model.generate_function, fake_generate_function)
            self.hook.assert_called_once_with(
                model, model.generate_step, static_argnames=static_argnames
            )
            # The function is cached. The hook is not called again.
            self.assertIs(
                model.make_generate_function(), fake_generate_function
            )
            self.assertEqual(self.hook.call_count, 1)
            # `compile()` clears the function. The hook builds a new one.
            model.compile()
            self.assertIs(
                model.make_generate_function(), fake_generate_function
            )
            self.assertEqual(self.hook.call_count, 2)

    def test_backend_hook_end_to_end_generate(self):
        calls = []

        def make_generate_function(model, generate_step, static_argnames=()):
            def generate_function(*args, **kwargs):
                calls.append((args, kwargs))
                return generate_step(*args, **kwargs)

            return generate_function

        self.hook.side_effect = make_generate_function
        backbone = GPT2Backbone(
            vocabulary_size=10,
            num_layers=1,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=8,
        )
        causal_lm = GPT2CausalLM(backbone=backbone, preprocessor=None)
        # A greedy sampler has no seed state, so `generate_step` can run
        # as is, without the state handling of a compiled function.
        causal_lm.compile(sampler="greedy")
        inputs = {
            "token_ids": np.array([[1, 2, 0, 0, 0, 0, 0, 0]]),
            "padding_mask": np.array([[1, 1, 0, 0, 0, 0, 0, 0]]),
        }
        with (
            backend_package("fakebackend", make_generate_function=self.hook),
            backend_name("fakebackend"),
        ):
            outputs = causal_lm.generate(inputs, stop_token_ids=None)
            self.hook.assert_called_once_with(
                causal_lm,
                causal_lm.generate_step,
                static_argnames=("stop_token_ids",),
            )
            # `generate()` ran through the function the hook returned.
            self.assertEqual(len(calls), 1)
            self.assertEqual(list(calls[0][1]), ["stop_token_ids"])
            self.assertEqual(outputs["token_ids"].shape, (1, 8))
            self.assertAllEqual(outputs["token_ids"][:, :2], [[1, 2]])
            # `compile()` clears the function. The hook builds a new one.
            causal_lm.compile(sampler="greedy")
            causal_lm.generate(inputs, stop_token_ids=None)
            self.assertEqual(self.hook.call_count, 2)
            self.assertEqual(len(calls), 2)

    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_backend_without_hook_is_unchanged(self, cls, static_argnames):
        # `src.keras_hub` is present and does not define the hook.
        model = cls()
        with backend_package("fakebackend"), backend_name("fakebackend"):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)
        # Top-level package is loaded without `src.keras_hub`.
        model = cls()
        with (
            backend_package(
                "fakebackend",
                with_keras_hub=False,
                make_generate_function=self.hook,
            ),
            backend_name("fakebackend"),
        ):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)
            self.hook.assert_not_called()
        # The backend package is not loaded.
        model = cls()
        with backend_name("fakebackend"):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)

    def test_helper_returns_without_mutating_model(self):
        model = CausalLM()
        custom_step = mock.Mock()
        with (
            backend_package("fakebackend", make_generate_function=self.hook),
            backend_name("fakebackend"),
        ):
            result = make_pluggable_backend_generate_function(
                model, custom_step, static_argnames=("stop_token_ids",)
            )
        self.assertIs(result, fake_generate_function)
        self.assertIsNone(model.generate_function)
        self.hook.assert_called_once_with(
            model, custom_step, static_argnames=("stop_token_ids",)
        )

        with backend_package("fakebackend"), backend_name("fakebackend"):
            result = make_pluggable_backend_generate_function(
                model, custom_step
            )
        self.assertIs(result, custom_step)
        self.assertIsNone(model.generate_function)

        with self.assertRaises(ValueError):
            make_pluggable_backend_generate_function(model, custom_step)


class BackendGenerateFunctionTest(_BackendGenerateFunctionTestCase):
    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_built_in_backend_ignores_backend_package(
        self, cls, static_argnames
    ):
        model = cls()
        backend = keras.config.backend()
        with backend_package(backend, make_generate_function=self.hook):
            generate_function = model.make_generate_function()
        self.hook.assert_not_called()
        self.assertIsNot(generate_function, fake_generate_function)

    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_numpy_backend_uses_generate_step(self, cls, static_argnames):
        model = cls()
        with (
            backend_package("numpy", make_generate_function=self.hook),
            backend_name("numpy", unofficial=False),
        ):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)
            self.assertEqual(model.generate_function, model.generate_step)
        self.hook.assert_not_called()

    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_eager_built_in_backend_uses_generate_step(
        self, cls, static_argnames
    ):
        model = cls()
        model.run_eagerly = True
        generate_function = model.make_generate_function()
        if keras.config.backend() in ("tensorflow", "jax"):
            self.assertEqual(generate_function, model.generate_step)
            self.assertEqual(model.generate_function, model.generate_step)
        else:
            self.assertTrue(callable(generate_function))
            self.assertIs(model.generate_function, generate_function)
