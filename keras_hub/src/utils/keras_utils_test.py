import sys
import types
from unittest import mock

import keras
import numpy as np
from absl.testing import parameterized

from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.image_to_image import ImageToImage
from keras_hub.src.models.inpaint import Inpaint
from keras_hub.src.models.text_to_image import TextToImage
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import clone_initializer


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


def backend_package(name, **attributes):
    """Put a `keras_<name>` backend package into `sys.modules`."""
    package = types.ModuleType(f"keras_{name}")
    package.__dict__.update(attributes)
    return mock.patch.dict(sys.modules, {package.__name__: package})


def backend_name(name):
    """Make `keras.config.backend()` return `name`."""
    return mock.patch.object(keras.config, "backend", return_value=name)


class BackendGenerateFunctionTest(TestCase):
    def setUp(self):
        super().setUp()
        built_in_backends = ("jax", "numpy", "tensorflow", "torch")
        if keras.config.backend() not in built_in_backends:
            self.skipTest("Simulates a pluggable backend on a built-in one.")
        self.hook = mock.Mock(return_value=fake_generate_function)

    @parameterized.named_parameters(*GENERATE_TASKS)
    def test_backend_hook_is_used(self, cls, static_argnames):
        model = cls()
        with (
            backend_package("fakebackend", make_generate_function=self.hook),
            backend_name("fakebackend"),
        ):
            generate_function = model.make_generate_function()
            self.assertIs(generate_function, fake_generate_function)
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
        # The backend package is loaded and does not define the hook.
        model = cls()
        with backend_package("fakebackend"), backend_name("fakebackend"):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)
        # The backend package is not loaded.
        model = cls()
        with backend_name("fakebackend"):
            generate_function = model.make_generate_function()
            self.assertEqual(generate_function, model.generate_step)

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
